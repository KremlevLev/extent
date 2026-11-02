"""Head-preserving PyTorch Mamba-3, no legacy checkpoints or JAX imports.

Kernel: state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa
Initialization adapted from arianraje's Apache-2.0 Qwen4B Mamba3 release
@b946069890837bbe9b5dd75e2d118a7e98d2d193. Gate Z remains explicitly random.
Projection ownership is preserved; softmax attention outputs are NOT preserved.
"""
import math
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F

SOURCE = 'Qwen/Qwen3-1.7B-Base'
REVISION = 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'
MAMBA_COMMIT = 'e9594ce1c732d97440f0332fdc43170a2294dbfa'
MASKS = {'gqa4': (6, 13, 20, 27), 'gqa7': (3, 7, 11, 15, 19, 23, 27)}


def rms(x, weight, eps):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (y * weight).to(x.dtype)


class CleanMamba(nn.Module):
    def __init__(self, attention, config, *, seed=123, rank=4, chunk=16, backend='chunked'):
        super().__init__()
        d, h, g, n = config.hidden_size, config.num_attention_heads, config.num_key_value_heads, config.head_dim
        if h * n != d or h % g or rank % (h // g):
            raise ValueError('incompatible donor heads; no pooling/cropping allowed')
        self.h, self.g, self.n, self.r, self.chunk = h, g, n, rank, chunk
        self.eps = config.rms_norm_eps
        self.backend = backend
        self.widths = [d, d, rank*g*n, rank*g*n, h, h, h, n//4]
        # Local CPU generator: surgery never changes global RNG used by data.
        gen = torch.Generator().manual_seed(seed)
        kernel = torch.zeros(sum(self.widths), d, dtype=torch.float32)
        pieces = kernel.split(self.widths)
        pieces[0].copy_(torch.randn(d, d, generator=gen) * .02)
        v = attention.v_proj.weight.detach().float().cpu().reshape(g, n, d)
        k = attention.k_proj.weight.detach().float().cpu().reshape(g, n, d)
        q = attention.q_proj.weight.detach().float().cpu().reshape(h, n, d)
        pieces[1].copy_(v.repeat_interleave(h//g, dim=0).reshape(d, d))
        pieces[2].copy_(k[None].expand(rank, -1, -1, -1).reshape(rank*g*n, d))
        pieces[3].copy_(torch.stack([q[torch.arange(g)*(h//g)+i%(h//g)] for i in range(rank)]).reshape(rank*g*n, d))
        self.in_weight = nn.Parameter(kernel)
        self.out_weight = nn.Parameter(attention.o_proj.weight.detach().float().cpu().clone())
        self.b_norm = nn.Parameter(attention.k_norm.weight.detach().float().cpu().clone())
        self.c_norm = nn.Parameter(attention.q_norm.weight.detach().float().cpu().clone())
        self.b_bias = nn.Parameter(torch.ones(h, rank, n))
        self.c_bias = nn.Parameter(torch.ones(h, rank, n))
        dt = torch.exp(torch.rand(h, generator=gen)*(math.log(.1)-math.log(.001))+math.log(.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.mimo_x = nn.Parameter(torch.randn(h, rank, n, generator=gen).softmax(1))
        own = torch.zeros(h, rank, n)
        own[torch.arange(h), torch.arange(h)%rank] = 1
        self.mimo_o = nn.Parameter(own)
        self.mimo_z = nn.Parameter(torch.ones(h, rank, n))
        self.D = nn.Parameter(torch.ones(h))
        self.readout_norm = nn.Parameter(torch.ones(h, n))
        self.verify_donor(attention)

    @torch.no_grad()
    def verify_donor(self, attention):
        z, x, b, c, *_ = self.in_weight.split(self.widths)
        q = attention.q_proj.weight.float().cpu().reshape(self.h, self.n, -1)
        k = attention.k_proj.weight.float().cpu().reshape(self.g, self.n, -1)
        v = attention.v_proj.weight.float().cpu().reshape(self.g, self.n, -1)
        c = c.cpu().reshape(self.r, self.g, self.n, -1)
        b = b.cpu().reshape(self.r, self.g, self.n, -1)
        for head in range(self.h):
            assert torch.equal(c[head%self.r, head//(self.h//self.g)], q[head])
        assert torch.equal(b, k[None].expand_as(b))
        assert torch.equal(x.cpu().reshape(self.h, self.n, -1), v.repeat_interleave(self.h//self.g, 0))
        assert torch.equal(self.out_weight.cpu(), attention.o_proj.weight.float().cpu())
        assert torch.equal(self.b_norm.cpu(), attention.k_norm.weight.float().cpu())
        assert torch.equal(self.c_norm.cpu(), attention.q_norm.weight.float().cpu())

    def components(self, u):
        z, x, b, c, dd, a, trap, angles = F.linear(u, self.in_weight).split(self.widths, dim=-1)
        batch, length = u.shape[:2]
        shape = (batch, length, self.h, self.n)
        b = rms(b.reshape(batch, length, self.r, self.g, self.n), self.b_norm, self.eps)
        c = rms(c.reshape(batch, length, self.r, self.g, self.n), self.c_norm, self.eps)
        dt = F.softplus(dd.float() + self.dt_bias)
        # Heavy-tail A, and raw angles: upstream angle_dt kernel applies tanh*pi.
        decay = -(a.float().clamp_min(0) + 1/(1-a.float().clamp_max(0)))
        return x.reshape(shape), z.reshape(shape), b, c, dt, decay.clamp_max(-1e-4), trap, angles

    def forward(self, u, *, reference=False):
        x, z, b, c, dt, decay, trap, angles = self.components(u)
        if reference:
            y = reference_scan(x, z, b, c, dt, decay, trap, angles, self)
        elif self.backend.startswith('chunked'):
            if not u.is_cuda:
                raise RuntimeError('Production requires CUDA fused/chunked MIMO; reference is test-only')
            y = chunked_scan(x, z, b, c, dt, decay, trap, angles, self, compiled=self.backend=='chunked-compiled')
        else:
            if not u.is_cuda:
                raise RuntimeError('Production requires CUDA fused MIMO; reference is test-only')
            from mamba_ssm.ops.tilelang.mamba3.mamba3_mimo import mamba3_mimo
            y = mamba3_mimo(Q=c, K=b, V=x, ADT=(dt*decay).transpose(1, 2),
                DT=dt.transpose(1, 2), Trap=trap.float().transpose(1, 2),
                Q_bias=self.c_bias, K_bias=self.b_bias, MIMO_V=self.mimo_x,
                MIMO_Z=self.mimo_z, MIMO_Out=self.mimo_o,
                Angles=angles.float().unsqueeze(2).expand(-1, -1, self.h, -1),
                D=self.D, Z=z, chunk_size=self.chunk, rotary_dim_divisor=4,
                dtype=x.dtype, fuse_pregate_headwise_rms_norm=True,
                outproj_norm_weight=self.readout_norm, outproj_norm_eps=1e-5)
        return F.linear(y.flatten(-2).to(x.dtype), self.out_weight)


def reference_scan(x, z, b, c, dt, decay, trap, angles, m):
    """Small diagnostic oracle; NEVER used as production training fallback."""
    batch, length, h, n = x.shape
    state = x.new_zeros((batch, h, n, n), dtype=torch.float32)
    pk = x.new_zeros((batch, m.r, h, n), dtype=torch.float32)
    pv = x.new_zeros((batch, h, n), dtype=torch.float32)
    phase = x.new_zeros((batch, h, n//4), dtype=torch.float32)
    bx = b.float().repeat_interleave(h//m.g, 3) + m.b_bias.transpose(0, 1)[None, None]
    cx = c.float().repeat_interleave(h//m.g, 3) + m.c_bias.transpose(0, 1)[None, None]
    def rotate(v, phase):
        cos, sin = phase.cos()[:, None], phase.sin()[:, None]
        cos = F.pad(cos, (0, n//2-n//4), value=1)
        sin = F.pad(sin, (0, n//2-n//4))
        a, b = v.chunk(2, -1)
        return torch.cat((a*cos-b*sin, b*cos+a*sin), -1)
    outputs = []
    for t in range(length):
        alpha = (dt[:, t]*decay[:, t]).exp()
        fraction = trap[:, t].float().sigmoid()
        gamma = fraction*dt[:, t]
        beta = (1-fraction)*dt[:, t]*alpha
        phase = phase + math.pi*angles[:, t, None].float().tanh()*dt[:, t, :, None]
        kb, qc = rotate(bx[:, t], phase), rotate(cx[:, t], phase)
        now = x[:, t, None].float()*m.mimo_x.transpose(0, 1)[None]
        prev = pv[:, None]*m.mimo_x.transpose(0, 1)[None]
        state = alpha[..., None, None]*state + torch.einsum('brhp,brhn->bhpn', gamma[:, None, :, None]*now, kb)
        state = state + torch.einsum('brhp,brhn->bhpn', beta[:, None, :, None]*prev, pk)
        read = torch.einsum('bhpn,brhn->brhp', state, qc) + m.D[None, None, :, None]*now
        read = rms(read, m.readout_norm[None, None], 1e-5)
        outputs.append((read*F.silu(z[:, t, None].float()*m.mimo_z.transpose(0, 1)[None])*m.mimo_o.transpose(0, 1)[None]).sum(1))
        pk, pv = kb, x[:, t].float()
    return torch.stack(outputs, 1).to(x.dtype)


def ssd_block(x, z, b, c, dt, decay, trap, angles, state, pk, pv, phase,
              b_bias, c_bias, mimo_x, mimo_z, mimo_o, skip, readout_weight):
    # Leave projection/MLP GEMMs in outer BF16 autocast. State algebra uses
    # FP32/TF32 to avoid loss from subtracting large cumulative log decays.
    with torch.autocast(device_type=x.device.type, enabled=False):
        batch, width, h, n = x.shape
        rank, groups = b.shape[2:4]
        alpha_log = dt.float()*decay.float()  # B,T,H
        prefix = alpha_log.cumsum(1).transpose(1, 2)  # B,H,T
        phase_all = phase[:, None] + (math.pi*angles.float().tanh()[:, :, None]*dt.float()[..., None]).cumsum(1)
        def rotate(v):
            cos = F.pad(phase_all.cos(), (0, n//2-n//4), value=1)[:, :, None]
            sin = F.pad(phase_all.sin(), (0, n//2-n//4))[:, :, None]
            a, b = v.chunk(2, -1)
            return torch.cat((a*cos-b*sin, b*cos+a*sin), -1)
        kb = rotate(b.float().repeat_interleave(h//groups, 3) + b_bias.transpose(0, 1)[None, None])
        qc = rotate(c.float().repeat_interleave(h//groups, 3) + c_bias.transpose(0, 1)[None, None])
        prevk = torch.cat((pk[:, None], kb[:, :-1]), 1)
        prevx = torch.cat((pv[:, None], x[:, :-1].float()), 1)
        now = x.float()[:, :, None]*mimo_x.transpose(0, 1)[None, None]
        prev = prevx[:, :, None]*mimo_x.transpose(0, 1)[None, None]
        gamma = trap.float().sigmoid()*dt.float()
        beta = (1-trap.float().sigmoid())*dt.float()*alpha_log.exp()
        # Flatten time/rank for efficient grouped GEMM. Cross-rank terms
        # matter: every input rank contributes to every output rank.
        def flat(t):
            return t.permute(0, 3, 1, 2, 4).reshape(batch, h, width*rank, n)
        q = flat(qc)
        triangular = torch.ones(width, width, device=x.device, dtype=torch.bool).tril()
        diff = prefix[..., :, None]-prefix[..., None, :]
        mask = diff.masked_fill(~triangular, -torch.inf).exp()
        def read(keys, values, coefficient):
            weights = mask*coefficient.transpose(1, 2)[..., None, :]
            weights = weights.repeat_interleave(rank, -2).repeat_interleave(rank, -1)
            score = torch.matmul(q, flat(keys).transpose(-1, -2))*weights
            return torch.matmul(score, flat(values))
        local = read(kb, now, gamma) + read(prevk, prev, beta)
        carry = torch.matmul(q, state.transpose(-1, -2))
        carry *= prefix.exp().repeat_interleave(rank, -1)[..., None]
        readout = (local+carry).reshape(batch, h, width, rank, n).permute(0, 2, 3, 1, 4)
        readout = readout + skip[None, None, None, :, None]*now
        readout = rms(readout, readout_weight[None, None, None], 1e-5)
        output = (readout*F.silu(z.float()[:, :, None]*mimo_z.transpose(0, 1)[None, None])*
            mimo_o.transpose(0, 1)[None, None]).sum(2)
        end_weights = (prefix[:, :, -1, None]-prefix).exp().transpose(1, 2)
        end = prefix[:, :, -1].exp()[..., None, None]*state
        for keys, values, coefficient in ((kb, now, gamma), (prevk, prev, beta)):
            weighted = keys*(end_weights*coefficient)[:, :, None, :, None]
            end = end + torch.matmul(flat(values).transpose(-1, -2), flat(weighted))
        return output.to(x.dtype), end, kb[:, -1], x[:, -1].float(), phase_all[:, -1]

@lru_cache(maxsize=1)
def compiled_ssd_block():
    return torch.compile(ssd_block, fullgraph=True, dynamic=False)


def chunked_scan(x, z, b, c, dt, decay, trap, angles, m, *, block=64, checkpoint=True, compiled=False):
    """Exact blockwise SSD algebra, bounded O(block^2 R^2) working memory.

    Carries S, previous K/V and rotary phase between chunks. FP32 recurrence and
    non-reentrant activation recomputation; no per-token Python loop, no custom
    shared-memory kernel requirement. Equivalent math, not a new architecture.
    """
    from torch.utils.checkpoint import checkpoint as recompute
    batch, length, h, n = x.shape
    state = x.new_zeros((batch, h, n, n), dtype=torch.float32)
    pk = x.new_zeros((batch, m.r, h, n), dtype=torch.float32)
    pv = x.new_zeros((batch, h, n), dtype=torch.float32)
    phase = x.new_zeros((batch, h, n//4), dtype=torch.float32)

    body = compiled_ssd_block() if compiled else ssd_block
    outputs = []
    for start in range(0, length, block):
        pieces = [t[:, start:start+block] for t in (x, z, b, c, dt, decay, trap, angles)]
        args = (*pieces, state, pk, pv, phase, m.b_bias, m.c_bias, m.mimo_x, m.mimo_z, m.mimo_o, m.D, m.readout_norm)
        if checkpoint and torch.is_grad_enabled():
            output, state, pk, pv, phase = recompute(body, *args, use_reentrant=False)
        else:
            output, state, pk, pv, phase = body(*args)
        outputs.append(output)
    return torch.cat(outputs, 1)


def microbatches(total, physical):
    if total < 1 or physical < 1:
        raise ValueError('positive batch sizes required')
    for offset in range(0, total, physical):
        yield offset, min(physical, total-offset)


def learning_rate(step, total=1526, warmup=46, peak=1e-3):
    if step < warmup:
        return peak*(step+1)/warmup
    fraction = min(1., (step-warmup)/max(1, total-warmup-1))
    return peak*(.1+.9*.5*(1+math.cos(math.pi*fraction)))


def replaced_block(layer, mixer, hidden, *, reference=False):
    middle = hidden + mixer(layer.input_layernorm(hidden), reference=reference)
    return middle + layer.mlp(layer.post_attention_layernorm(middle))


def layer_loss(prediction, target):
    # Per-sequence target energy: invariant to physical microbatch partition.
    error = (prediction.float()-target.float()).square().mean((1, 2))
    energy = target.float().square().mean((1, 2)).detach().clamp_min(1e-8)
    return (error/energy).mean()

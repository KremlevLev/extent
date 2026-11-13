"""Full CE student and exact Optax-to-Torch state mapping for EXP176."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from extent.broad_corpus import digest_file
from extent.quality_172 import MAPPING
from extent.torch_clean_170 import CleanMamba, MASKS, chunked_scan, rms


class ExactRMSNorm(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.weight = original.weight
        self.eps = original.variance_epsilon

    def forward(self, inputs):
        return rms(inputs, self.weight, self.eps)


class TrainingMamba(CleanMamba):
    def forward(self, inputs, *, reference=False):
        reference = reference or self.backend == "reference"
        if reference or not self.backend.startswith("chunked"):
            return super().forward(inputs, reference=reference)
        values = self.components(inputs)
        output = chunked_scan(
            *values, self, checkpoint=False, compiled=self.backend == "chunked-compiled",
        )
        return F.linear(output.flatten(-2).to(inputs.dtype), self.out_weight)


def placeholder_attention(config):
    """Shape-only donor for allocating modules before strict checkpoint loading."""
    d, n, g = config.hidden_size, config.head_dim, config.num_key_value_heads
    return SimpleNamespace(
        q_proj=SimpleNamespace(weight=torch.zeros(d, d)),
        k_proj=SimpleNamespace(weight=torch.zeros(g * n, d)),
        v_proj=SimpleNamespace(weight=torch.zeros(g * n, d)),
        o_proj=SimpleNamespace(weight=torch.zeros(d, d)),
        q_norm=SimpleNamespace(weight=torch.ones(n)), k_norm=SimpleNamespace(weight=torch.ones(n)),
    )


def ce_tile(features, head, targets):
    return F.cross_entropy(F.linear(features, head).float(), targets, reduction="sum")

def sample_master_weights(model):
    names = ("qwen.embed_tokens.weight", "qwen.norm.weight",
             "qwen.layers.3.self_attn.q_proj.weight", "mixers.0.in_weight")
    parameters = dict(model.named_parameters())
    result = {}
    for name in names:
        flat = parameters[name].detach().reshape(-1)
        indices = torch.linspace(0, len(flat) - 1, min(1024, len(flat)), device=flat.device).long()
        result[name] = flat[indices].float().cpu()
    return result


def host_precision_changes(before, after):
    result = {}
    for name, left in before.items():
        right = after[name]
        # CPU materialized BF16 storage bits, independent of accelerator compare fusion.
        old_bits = left.to(torch.bfloat16).view(torch.uint16)
        new_bits = right.to(torch.bfloat16).view(torch.uint16)
        result[name] = dict(samples=left.numel(),
            fp32_changed_fraction=float((left != right).float().mean()),
            bf16_changed_fraction=float((old_bits != new_bits).float().mean()),
            max_abs_change=float((left - right).abs().max()))
    return result


class FullStudent(nn.Module):
    def __init__(self, config, *, backend="chunked-compiled"):
        super().__init__()
        from transformers import AutoModel
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

        with torch.device("meta"):
            self.qwen = AutoModel.from_config(config, attn_implementation="sdpa")
        self.qwen.embed_tokens.padding_idx = None
        # Nonpersistent rotary buffers are not restored by state_dict.
        self.qwen.rotary_emb = Qwen3RotaryEmbedding(config, device="cpu")
        for module in list(self.qwen.modules()):
            for name, child in list(module.named_children()):
                if child.__class__.__name__ == "Qwen3RMSNorm":
                    setattr(module, name, ExactRMSNorm(child))
        self.retained = set(MASKS["gqa7"])
        donor = placeholder_attention(config)
        self.mixers = nn.ModuleDict()
        for index, layer in enumerate(self.qwen.layers):
            if index not in self.retained:
                self.mixers[str(index)] = TrainingMamba(donor, config, backend=backend)
                layer.self_attn = nn.Identity()
        self.checkpoint_blocks = True

    @staticmethod
    def block(hidden, layer, mixer, positions, rope):
        if mixer is None:
            return layer(hidden, position_ids=positions, position_embeddings=rope, use_cache=False)
        middle = hidden + mixer(layer.input_layernorm(hidden))
        return middle + layer.mlp(layer.post_attention_layernorm(middle))

    def hidden(self, tokens):
        hidden = self.qwen.embed_tokens(tokens)
        if tokens.is_cuda:
            hidden = hidden.to(torch.bfloat16)
        positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
        rope = self.qwen.rotary_emb(hidden, positions)
        for index, layer in enumerate(self.qwen.layers):
            mixer = None if index in self.retained else self.mixers[str(index)]
            args = (hidden, layer, mixer, positions, rope)
            if self.checkpoint_blocks and torch.is_grad_enabled():
                hidden = checkpoint(self.block, *args, use_reentrant=False)
            else:
                hidden = self.block(*args)
        return self.qwen.norm(hidden)

    def forward(self, tokens):
        hidden = self.hidden(tokens)[:, :-1].reshape(-1, self.qwen.config.hidden_size)
        targets = tokens[:, 1:].reshape(-1)
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(targets), 128):
            args = (hidden[start:start + 128], self.qwen.embed_tokens.weight, targets[start:start + 128])
            value = checkpoint(ce_tile, *args, use_reentrant=False) if torch.is_grad_enabled() else ce_tile(*args)
            total = total + value
        return total / targets.numel()


def destinations(path, layers=28):
    """Yield Torch name, source layer index and transpose flag for one JAX leaf."""
    if path[0] == "mamba":
        name, transpose = MAPPING[tuple(path[1:])]
        for index in range(layers - 1):
            if index not in MASKS["gqa7"]:
                yield f"mixers.{index}.{name}", index, transpose
        return
    if path[0] != "qwen":
        raise ValueError("Unknown source parameter group")
    group, *parts = path[1:]
    if group == "2":
        yield "qwen.embed_tokens.weight", None, False
    elif group == "3":
        if parts != ["scale"]:
            raise ValueError("Invalid final norm path")
        yield "qwen.norm.weight", None, False
    elif group in ("0", "1"):
        transpose = parts[-1] == "kernel"
        if parts[-1] not in ("kernel", "scale"):
            raise ValueError("Invalid Qwen tensor leaf")
        suffix = ".".join(parts[:-1] + ["weight"])
        indices = range(layers - 1) if group == "0" else [layers - 1]
        for index in indices:
            if parts[0] == "self_attn" and index not in MASKS["gqa7"]:
                continue
            yield f"qwen.layers.{index}.{suffix}", index if group == "0" else None, transpose
    else:
        raise ValueError("Unknown Qwen tuple entry")


def read_row(row, directory):
    flat = np.empty(int(np.prod(row["shape"])), dtype=row["dtype"])
    at = 0
    for chunk in row["chunks"]:
        name = Path(chunk["file"])
        if name.is_absolute() or ".." in name.parts or "\\" in chunk["file"]:
            raise ValueError("Unsafe source chunk path")
        file = directory / name
        if digest_file(file) != chunk["sha256"]:
            raise ValueError("Source chunk SHA mismatch")
        values = np.load(file, allow_pickle=False)
        if values.dtype != flat.dtype or at + values.size > flat.size:
            raise ValueError("Invalid source chunk")
        flat[at:at + values.size] = values.reshape(-1)
        at += values.size
    if at != flat.size:
        raise ValueError("Truncated source leaf")
    return flat.reshape(row["shape"])


def convert_source(model, metadata, directory):
    mapped = {name: {} for name in ("parameters", "mu", "nu")}
    count, cursor = None, None
    for row in metadata["leaves"]:
        path = row["path"]
        array = read_row(row, directory)
        if path == ["optimizer", "1", "count"]:
            count = int(array)
            continue
        if path == ["cursor"]:
            cursor = int(array)
            continue
        if path[0] == "parameters":
            group, leaf = "parameters", path[1:]
        elif path[:2] == ["optimizer", "1"] and path[2] in ("mu", "nu"):
            group, leaf = path[2], path[3:]
        else:
            raise ValueError("Unexpected checkpoint leaf")
        for name, index, transpose in destinations(leaf, model.qwen.config.num_hidden_layers):
            if name in mapped[group]:
                raise ValueError("Duplicate mapped parameter")
            value = array if index is None else array[index]
            if transpose:
                value = value.T
            mapped[group][name] = torch.from_numpy(np.array(value, dtype=np.float32, order="C", copy=True))
    expected = dict(model.named_parameters())
    for group, values in mapped.items():
        if values.keys() != expected.keys():
            raise ValueError(f"Incomplete {group} mapping")
        if any(values[name].shape != reference.shape for name, reference in expected.items()):
            raise ValueError(f"Wrong {group} shape")
    if count != metadata["step"] or cursor != 83886080 + count * 16384:
        raise ValueError("Source Adam count or cursor mismatch")
    model.load_state_dict(mapped["parameters"], strict=True, assign=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(.9, .95), eps=1e-8, weight_decay=0.)
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = dict(
            step=torch.tensor(float(count)), exp_avg=mapped["mu"][name], exp_avg_sq=mapped["nu"][name],
        )
    return optimizer, cursor


def save_current(directory, payload):
    from filelock import FileLock
    from extent.broad_corpus import write_json

    target = directory / "CURRENT.pt"
    temporary = directory / "CURRENT.next.pt"
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    sha = digest_file(temporary)
    with FileLock(str(directory / "current.lock")):
        temporary.replace(target)
        write_json(directory / "CURRENT.json", dict(
            step=payload["step"], cursor=payload["cursor"], sha256=sha,
            contract=payload["contract"], bytes=target.stat().st_size,
        ))
    return target


def load_config(directory):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(directory, local_files_only=True)
    if (config.num_hidden_layers, config.hidden_size, config.intermediate_size,
            config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.vocab_size, config.tie_word_embeddings) != (
            28, 2048, 6144, 16, 8, 128, 151936, True):
        raise ValueError("Pinned Qwen3-1.7B geometry differs")
    return config

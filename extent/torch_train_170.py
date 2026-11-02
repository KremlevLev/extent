"""EXP170 clean stage-1 alignment. PyTorch only; shared frozen teacher pass.

Every original teacher block produces one target. Its replacement is immediately
backpropagated, so no 27-layer activation bank or duplicated teacher is stored.
One globally clipped AdamW update after all16 sequences/all27 mixers.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from extent.torch_clean_170 import (CleanMamba, MASKS, SOURCE, REVISION,
    MAMBA_COMMIT, microbatches, learning_rate, layer_loss, replaced_block)
from extent.broad_corpus import PackedCorpus, digest_file


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def tensor_hash(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def sync():
    torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.values = {}

    @contextlib.contextmanager
    def phase(self, name):
        # CUDA events measure device time without synchronizing each layer.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.values.setdefault(name, []).append((start, end))

    def finish(self):
        sync()
        result = {name: sum(a.elapsed_time(b) for a, b in events)/1000 for name, events in self.values.items()}
        self.values.clear()
        return result


class Alignment:
    def __init__(self, args):
        import transformers
        from transformers import AutoModel
        if transformers.__version__ != '4.57.1':
            raise ValueError('install pinned transformers4.57.1 in the separate PyTorch environment')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA with native BF16 required')
        torch.manual_seed(170)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.set_per_process_memory_fraction(args.memory_gib*1024**3/torch.cuda.get_device_properties(0).total_memory)
        self.teacher = AutoModel.from_pretrained(SOURCE, revision=REVISION,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            cache_dir=args.model_cache, trust_remote_code=False).eval().to('cuda')
        self.teacher.requires_grad_(False)
        if (self.teacher.config.num_hidden_layers, self.teacher.config.hidden_size,
            self.teacher.config.num_attention_heads, self.teacher.config.num_key_value_heads,
            self.teacher.config.head_dim) != (28, 2048, 16, 8, 128):
            raise ValueError('pinned Qwen geometry differs')
        if not self.teacher.config.tie_word_embeddings:
            raise ValueError('evaluation requires pinned tied embedding/head')
        self.bank = torch.nn.ModuleDict({str(i): CleanMamba(layer.self_attn,
            self.teacher.config, seed=123+i, chunk=args.chunk, backend=args.backend) for i, layer in enumerate(self.teacher.layers[:-1])}).cuda()
        self.optimizer = torch.optim.AdamW(self.bank.parameters(), lr=1e-3,
            betas=(.9, .95), eps=1e-8, weight_decay=0., fused=True)
        self.corpus = PackedCorpus(args.corpus)
        contract = self.corpus.manifest['contract']
        if contract.get('tokenizer') != SOURCE or contract.get('tokenizer_revision') != REVISION:
            raise ValueError('corpus tokenizer differs from original Qwen')
        if int(self.corpus.arrays['train'].min()) < 0 or int(self.corpus.arrays['train'].max()) >= self.teacher.config.vocab_size:
            raise ValueError('corpus token ids outside vocabulary')
        self.length, self.effective = args.sequence_length, args.effective_batch
        self.stop_dir = Path(args.work_dir)
        if args.eval_windows*self.length > len(self.corpus.arrays['validation']):
            raise ValueError('not enough declared validation tokens')
        self.audit = {str(i): {name: tensor_hash(getattr(layer.self_attn, name).weight)
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm')}
            for i, layer in enumerate(self.teacher.layers[:-1])}
        self.contract = dict(experiment=170, format='torch-clean-align-v1', source=SOURCE,
            revision=REVISION, mamba_commit=MAMBA_COMMIT, corpus_identity=self.corpus.identity,
            teacher_implementation='transformers4.57.1-sdpa',
            sequence_length=self.length, effective_batch=self.effective, total_updates=args.total_updates,
            initializer='QKVO-own-head; Z=random; dynamics=zero', seed=123,
            geometry=dict(H=16, G=8, P=128, N=128, R=4),
            target='teacher-block-output; per-sequence relative MSE averaged over27 layers',
            optimizer=dict(name='AdamW', betas=[.9, .95], eps=1e-8, wd=0, clip=1,
                peak_lr=1e-3, warmup=46, cosine_final_fraction=.1), donor_hashes=self.audit,
            implementation_hashes={name: digest_file(Path(__file__).parent/name)
                for name in ('torch_clean_170.py', 'torch_train_170.py')})
        self.step, self.cursor = 0, 0
        self.timer = Timer()

    def batch(self, cursor, count, split='train'):
        start, end = cursor, cursor+count*self.length
        values = self.corpus.arrays[split]
        if end > len(values):
            raise EOFError('fresh corpus exhausted; no cycling')
        return torch.from_numpy(np.array(values[start:end], dtype=np.int64).reshape(count, self.length)).pin_memory().to('cuda', non_blocking=True)

    def positions(self, hidden):
        position = torch.arange(hidden.shape[1], device=hidden.device)[None]
        return position, self.teacher.rotary_emb(hidden, position)

    def update(self, physical, *, lr, trace=False):
        self.optimizer.zero_grad(set_to_none=True)
        device = next(self.bank.parameters()).device
        loss_value = torch.zeros((), device=device)
        local = []
        for offset, count in microbatches(self.effective, physical):
            with self.timer.phase('input_transfer'):
                tokens = self.batch(self.cursor+offset*self.length, count)
            with torch.no_grad(), self.timer.phase('teacher_embedding'):
                hidden = self.teacher.embed_tokens(tokens)
                positions, rope = self.positions(hidden)
            for index, layer in enumerate(self.teacher.layers[:-1]):
                teacher_mixer, student_mixer = [], []
                hooks = []
                if trace and str(index) in self.bank:
                    hooks.append(layer.self_attn.register_forward_hook(lambda mod, inp, out: teacher_mixer.append(out[0].detach())))
                    hooks.append(self.bank[str(index)].register_forward_hook(lambda mod, inp, out: student_mixer.append(out.detach())))
                with torch.no_grad(), self.timer.phase('teacher_blocks'):
                    target = layer(hidden, position_ids=positions, position_embeddings=rope, use_cache=False)
                if str(index) in self.bank:
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'), self.timer.phase('student_forward'):
                        prediction = replaced_block(layer, self.bank[str(index)], hidden)
                        loss = layer_loss(prediction, target)
                    with self.timer.phase('student_backward'):
                        (loss*(count/self.effective)/len(self.bank)).backward()
                    loss_value += loss.detach()*(count/self.effective)/len(self.bank)
                    if trace:
                        local.append({'layer': index, 'relative_mse': float(loss.detach()),
                            'mixer_relative_mse': float(layer_loss(student_mixer[0], teacher_mixer[0])),
                            'teacher_rms': float(target.float().square().mean().sqrt()),
                            'prediction_rms': float(prediction.detach().float().square().mean().sqrt())})
                    del prediction, loss
                for hook in hooks:
                    hook.remove()
                hidden = target
        with self.timer.phase('clip_optimizer'):
            if not torch.isfinite(loss_value):
                raise FloatingPointError('nonfinite alignment loss; optimizer not advanced')
            norm = torch.nn.utils.clip_grad_norm_(self.bank.parameters(), 1., error_if_nonfinite=True)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            self.optimizer.step()
        metrics = self.timer.finish()
        self.step += 1
        self.cursor += self.effective*self.length
        return dict(loss=float(loss_value), grad_norm=float(norm), lr=lr, phases_seconds=metrics, layers=local)

    def kernel_gate(self):
        """One real layer, real weights, nonzero angles; fused vs recurrence AD."""
        module = self.bank['0']
        gen = torch.Generator(device='cuda').manual_seed(170)
        inputs = torch.randn(1, 32, 2048, device='cuda', dtype=torch.bfloat16, generator=gen)*.2
        # Test nonzero dynamics as well as initialization, then restore exactly.
        with torch.no_grad():
            original = module.in_weight[-module.n//4:].clone()
            module.in_weight[-module.n//4:].normal_(0, .001, generator=gen)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                fused = module(inputs)
                grad_f = torch.autograd.grad(fused.float().square().mean(), (module.in_weight, module.out_weight))
                reference = module(inputs, reference=True)
                grad_r = torch.autograd.grad(reference.float().square().mean(), (module.in_weight, module.out_weight))
            output_error = float((fused.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
            errors = [float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-8)) for a, b in zip(grad_f, grad_r)]
            cosines = [float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0)) for a, b in zip(grad_f, grad_r)]
            if output_error > .03 or max(errors) > .15 or min(cosines) < .98:
                raise ValueError('fused/reference numerical gate failed; do not relax automatically')
            return dict(output_relative_l2=output_error, gradient_relative_l2=errors, gradient_cosine=cosines)
        finally:
            with torch.no_grad():
                module.in_weight[-module.n//4:].copy_(original)

    @torch.no_grad()
    def evaluate(self, windows=8):
        results = {}
        # All three use the same held-out Edu bytes/context, NEVER relabel A-E.
        for name, retained in [('teacher', tuple(range(28))), *MASKS.items()]:
            total, targets = 0., 0
            for window in range(windows):
                if (self.stop_dir/'FAST_STOP').exists():
                    return None  # Discard incomplete panels; do not compare unequal windows.
                tokens = self.batch(window*self.length, 1, split='validation')
                hidden = self.teacher.embed_tokens(tokens)
                positions, rope = self.positions(hidden)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for index, layer in enumerate(self.teacher.layers):
                        if index in retained:
                            hidden = layer(hidden, position_ids=positions, position_embeddings=rope, use_cache=False)
                        else:
                            hidden = replaced_block(layer, self.bank[str(index)], hidden)
                    hidden = self.teacher.norm(hidden)
                    # Avoid a full B*4096*151936 logits tensor.
                    for start in range(0, self.length-1, 128):
                        end = min(self.length-1, start+128)
                        logits = F.linear(hidden[:, start:end], self.teacher.embed_tokens.weight).float()
                        total += float(F.cross_entropy(logits.flatten(0, 1), tokens[:, start+1:end+1].flatten(), reduction='sum'))
                        targets += end-start
            results[name+'_edu_validation_nll'] = total/targets
        return results

    def profile_first_layer(self, physical, root):
        """Bounded actual-context trace, no optimizer update, no data advance."""
        tokens = self.batch(self.cursor, physical)
        with torch.no_grad():
            hidden = self.teacher.embed_tokens(tokens)
            positions, rope = self.positions(hidden)
        layer = self.teacher.layers[0]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA], with_flops=True) as profile:
            with torch.no_grad(), torch.profiler.record_function('teacher_block'):
                target = layer(hidden, position_ids=positions, position_embeddings=rope, use_cache=False)
            with torch.autocast('cuda', dtype=torch.bfloat16), torch.profiler.record_function('student_block'):
                loss = layer_loss(replaced_block(layer, self.bank['0'], hidden), target)
            with torch.profiler.record_function('student_backward'):
                loss.backward()
            sync()
        self.optimizer.zero_grad(set_to_none=True)
        profile.export_chrome_trace(str(root/'profile-layer0.json'))
        (root/'profile-layer0.txt').write_text(profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
        return dict(scope='layer0 teacher/student/backward; actual context and physical batch; no optimizer update',
            engineering_input_presentations=physical*self.length,
            counted_operator_flops=sum(e.flops for e in profile.key_averages()),
            flop_limitation='profiler does not count all operators; not measured MFU')


def save_checkpoint(engine, root):
    root = Path(root)
    # One atomic CURRENT; no original teacher/embedding copies in checkpoints.
    path, temporary = root/'CURRENT.pt', root/'CURRENT.tmp'
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError('refusing symlink checkpoint paths')
    required = sum(p.numel()*p.element_size() for p in engine.bank.parameters())*4 + 1024**3
    if shutil.disk_usage(root).free < required:
        raise OSError('insufficient disk for atomic CURRENT; previous checkpoint retained')
    payload = dict(contract=engine.contract, step=engine.step, cursor=engine.cursor,
        bank=engine.bank.state_dict(), optimizer=engine.optimizer.state_dict(),
        branch=getattr(engine, 'report_branch', None))
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def restore_checkpoint(engine, path):
    # Own tensor/dict format only; never execute pickle globals from a repository.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['contract'] != engine.contract:
        raise ValueError('CURRENT contract differs; refusing silent restart or changed corpus')
    step, cursor = payload['step'], payload['cursor']
    if cursor != step*engine.effective*engine.length:
        raise ValueError('optimizer/data cursor disagree')
    engine.bank.load_state_dict(payload['bank'], strict=True)
    engine.optimizer.load_state_dict(payload['optimizer'])
    engine.step, engine.cursor = step, cursor
    engine.report_branch = payload.get('branch')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--model-cache', required=True)
    p.add_argument('--memory-gib', type=float, default=85.)
    p.add_argument('--sequence-length', type=int, default=4096, choices=[256, 1024, 2048, 4096])
    p.add_argument('--effective-batch', type=int, default=16)
    p.add_argument('--total-updates', type=int, default=1526)
    p.add_argument('--chunk', type=int, default=16, choices=[8, 16])
    p.add_argument('--backend', choices=['chunked-compiled', 'chunked', 'tilelang'], default='chunked')
    p.add_argument('--physical-batch', type=int, default=1)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--result', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--save-reserve-minutes', type=float, default=18.)
    p.add_argument('--eval-every', type=int, default=128)
    p.add_argument('--eval-windows', type=int, default=8)
    p.add_argument('--profile', action='store_true')
    return p


def worker(argv=None):
    args = parser().parse_args(argv)
    root = Path(args.work_dir)
    result = dict(status='initializing', contract={'experiment':170})
    write(args.result, result)
    started = time.monotonic()
    try:
        engine = Alignment(args)
        result['contract'] = engine.contract
        result['execution'] = dict(backend=args.backend, chunk=args.chunk,
            torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            shared_memory_per_block=torch.cuda.get_device_properties(0).shared_memory_per_block,
            trainable_parameters=sum(p.numel() for p in engine.bank.parameters()),
            composed_parameters={name:sum(p.numel() for p in engine.teacher.parameters()) +
                sum(sum(p.numel() for p in engine.bank[str(i)].parameters()) -
                    sum(p.numel() for p in layer.self_attn.parameters())
                    for i,layer in enumerate(engine.teacher.layers) if i not in retained)
                for name,retained in MASKS.items()})
        result['kernel_gate'] = engine.kernel_gate()
        if args.probe:
            timings, phase = [], []
            # First update includes Adam allocation + compilation; two timed updates.
            for index in range(3):
                sync(); begin = time.monotonic()
                row = engine.update(args.physical_batch, lr=1e-3)
                sync(); elapsed = time.monotonic()-begin
                if index:
                    timings.append(elapsed); phase.append(row['phases_seconds'])
            result.update(status='measured', seconds_per_effective_update=float(np.median(timings)),
                input_tokens_per_second=engine.length*engine.effective/float(np.median(timings)),
                phases=phase, max_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_reserved_bytes=torch.cuda.max_memory_reserved(), accepted_updates=0, finite=True,
                engineering_input_presentations=3*engine.length*engine.effective)
            # Probe updates are discarded and have no scientific checkpoint.
            return
        current = root/'CURRENT.pt'
        if current.exists():
            restore_checkpoint(engine, current)
        origin_cursor = engine.cursor
        branch = getattr(engine, 'report_branch', None) or dict(step=engine.step, source_step=0,
            accepted_input_tokens=engine.cursor, history=[], evaluations=[])
        branch['selected_batch'] = args.physical_batch
        engine.report_branch = branch
        result['branches'] = {'CLEAN-ALIGN':branch}
        result['status'] = 'running'; write(args.result, result)
        if args.profile and time.time() < args.deadline-args.save_reserve_minutes*60:
            result['profiler'] = engine.profile_first_layer(args.physical_batch, root)
        if not any((root/leaf).exists() for leaf in ('STOP','FAST_STOP')) and time.time() < args.deadline-args.save_reserve_minutes*60:
            begin = time.monotonic()
            evaluation = engine.evaluate(args.eval_windows)
            if evaluation is not None:
                branch['evaluations'].append(dict(step=engine.step, **evaluation))
            result['last_eval_seconds'] = time.monotonic()-begin
        last_save = time.monotonic()
        milestones = {math.ceil(tokens/(engine.effective*engine.length)) for tokens in (1_000_000,2_000_000,5_000_000,10_000_000,25_000_000,50_000_000,100_000_000)}
        while engine.step < args.total_updates:
            if (root/'STOP').exists() or (root/'FAST_STOP').exists():
                result['status'] = 'fast_paused' if (root/'FAST_STOP').exists() else 'stopped'
                break
            if time.time() >= args.deadline-args.save_reserve_minutes*60:
                result['status'] = 'deadline_partial'; break
            if engine.cursor+engine.effective*engine.length > len(engine.corpus.arrays['train']):
                result['status'] = 'corpus_exhausted'; break
            sync(); begin = time.monotonic()
            row = engine.update(args.physical_batch, lr=learning_rate(engine.step, args.total_updates), trace=engine.step%128==0)
            elapsed = time.monotonic()-begin
            row.update(step=engine.step, seconds=elapsed, input_tokens=engine.effective*engine.length,
                accepted_input_tokens=engine.cursor, unix_time=time.time(),
                input_tokens_per_second=engine.effective*engine.length/elapsed)
            branch.update(step=engine.step, accepted_input_tokens=engine.cursor)
            branch['history'].append(row)
            result.update(launcher_elapsed_hours=(time.monotonic()-started)/3600,
                accepted_this_invocation_tokens=engine.cursor-origin_cursor,
                training_tokens_per_second=(engine.cursor-origin_cursor)/max(1.,time.monotonic()-started),
                max_allocated_bytes=torch.cuda.max_memory_allocated())
            write(args.result, result)
            if (engine.step%args.eval_every == 0 or engine.step in milestones) and time.time()+2*result.get('last_eval_seconds', 60) < args.deadline-args.save_reserve_minutes*60:
                begin = time.monotonic()
                evaluation = engine.evaluate(args.eval_windows)
                if evaluation is not None:
                    branch['evaluations'].append(dict(step=engine.step, **evaluation))
                result['last_eval_seconds'] = time.monotonic()-begin
            if time.monotonic()-last_save > 3600:
                begin = time.monotonic(); save_checkpoint(engine, root)
                result['last_local_save_seconds'] = time.monotonic()-begin
                last_save = time.monotonic()
        else:
            result['status'] = 'completed'
        save_checkpoint(engine, root)
        result['local_checkpoint_step'] = engine.step
        if (root/'FAST_STOP').exists():
            result['status'] = 'fast_paused'
        if result['status'] != 'fast_paused' and time.time()+result.get('last_eval_seconds', 60) < args.deadline-60:
            evaluation = engine.evaluate(args.eval_windows)
            if evaluation is not None:
                branch['evaluations'].append(dict(step=engine.step, **evaluation))
        if (root/'FAST_STOP').exists():
            result['status'] = 'fast_paused'
    except torch.cuda.OutOfMemoryError:
        result.update(status='memory_limit', error_type='OutOfMemoryError')
        if not args.probe:
            raise
    except BaseException as exc:
        result.update(status='failed', error_type=type(exc).__name__)
        if not args.probe and 'engine' in locals() and engine.step > 0:
            try:
                save_checkpoint(engine, root)
                result['local_checkpoint_step'] = engine.step
            except Exception as save_error:
                result['emergency_save_error'] = type(save_error).__name__
        raise
    finally:
        result['worker_elapsed_seconds'] = time.monotonic()-started
        write(args.result, result)


if __name__ == '__main__':
    worker()

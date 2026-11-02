import contextlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from extent.torch_clean_170 import CleanMamba, layer_loss, learning_rate, microbatches, replaced_block, chunked_scan
from extent.torch_train_170 import save_checkpoint, restore_checkpoint


def donor():
    torch.manual_seed(9)
    config = SimpleNamespace(hidden_size=16, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, rms_norm_eps=1e-6)
    attn = torch.nn.Module()
    for name, outputs in [('q_proj', 16), ('k_proj', 8), ('v_proj', 8), ('o_proj', 16)]:
        setattr(attn, name, torch.nn.Linear(16, outputs, bias=False))
    attn.q_norm = torch.nn.LayerNorm(4, bias=False)
    attn.k_norm = torch.nn.LayerNorm(4, bias=False)
    return config, attn


def test_head_ownership_dynamic_rows_zero_and_original_unchanged():
    config, attn = donor()
    original = {k: v.clone() for k, v in attn.state_dict().items()}
    m = CleanMamba(attn, config)
    m.verify_donor(attn)
    assert torch.count_nonzero(m.in_weight[-(3*4+1):]) == 0
    assert all(torch.equal(v, original[k]) for k, v in attn.state_dict().items())
    assert torch.allclose(m.mimo_x.sum(1), torch.ones(4, 4))
    assert torch.equal(m.mimo_o.sum(1), torch.ones(4, 4))
    with torch.no_grad():
        m.in_weight[16] += 1
    with pytest.raises(AssertionError):
        m.verify_donor(attn)


def test_no_silent_pooling_and_no_cpu_production_fallback():
    config, attn = donor()
    config.hidden_size = 15
    with pytest.raises(ValueError, match='no pooling'):
        CleanMamba(attn, config)
    config.hidden_size = 16
    m = CleanMamba(attn, config)
    with pytest.raises(RuntimeError, match='CUDA fused'):
        m(torch.randn(1, 4, 16))


def test_torch_reference_matches_existing_jax_with_nonzero_angles_and_gradients():
    import numpy as np
    import jax
    import jax.numpy as jnp
    from extent.config import Mamba3Config
    from extent.clean_transplant_170 import forward
    config, attn = donor()
    m = CleanMamba(attn, config)
    with torch.no_grad():
        m.in_weight[-1].normal_(0, .03)
        m.in_weight[-13:-1].normal_(0, .01)
    p = dict(in_proj={'kernel':m.in_weight.detach().numpy().T},
        out_proj={'kernel':m.out_weight.detach().numpy().T},
        b_norm={'scale':m.b_norm.detach().numpy()}, c_norm={'scale':m.c_norm.detach().numpy()},
        readout_norm={'scale':m.readout_norm.detach().numpy()})
    for name in ('b_bias', 'c_bias', 'dt_bias', 'mimo_x', 'mimo_z', 'mimo_o', 'D'):
        p[name] = getattr(m, name).detach().numpy()
    cfg = SimpleNamespace(hidden_size=16, mamba=Mamba3Config(head_dim=4, d_state=4, groups=2, expand=1., mimo_rank=4),
        compute_dtype='float32', rms_norm_eps=1e-6)
    inputs = torch.randn(2, 9, 16)*.2
    actual = m(inputs, reference=True)
    expected = forward(jnp.asarray(inputs.numpy()), p, cfg)
    np.testing.assert_allclose(actual.detach().numpy(), np.asarray(expected), rtol=2e-4, atol=2e-5)
    actual.square().mean().backward()
    gradient = jax.grad(lambda q: jnp.mean(forward(jnp.asarray(inputs.numpy()), q, cfg)**2))(p)
    np.testing.assert_allclose(m.in_weight.grad.numpy().T, np.asarray(gradient['in_proj']['kernel']), rtol=2e-3, atol=2e-6)
    assert torch.isfinite(m.in_weight.grad).all()
    assert m.in_weight.grad[-1].abs().sum() > 0


def test_relative_loss_weighted_microbatches_same_gradient():
    inputs = torch.randn(16, 5, 7)
    targets = torch.randn_like(inputs)
    weight = torch.tensor(.8, requires_grad=True)
    layer_loss(inputs*weight, targets).backward()
    full = weight.grad.clone(); weight.grad = None
    assert list(microbatches(16, 6)) == [(0, 6), (6, 6), (12, 4)]
    for start, count in microbatches(16, 6):
        (layer_loss(inputs[start:start+count]*weight, targets[start:start+count])*count/16).backward()
    torch.testing.assert_close(weight.grad, full)


@pytest.mark.parametrize('checkpoint,compiled', [(False,False),(True,False),(True,True)])
def test_blockwise_ssd_matches_recurrence_outputs_and_all_parameter_gradients(checkpoint, compiled, monkeypatch):
    if compiled:
        import extent.torch_clean_170 as core
        # CPU AOT capture/backward validates fullgraph compatibility; CUDA
        # Inductor codegen is separately tested by the real-device preflight.
        captured = torch.compile(core.ssd_block, backend='aot_eager', fullgraph=True, dynamic=False)
        monkeypatch.setattr(core, 'compiled_ssd_block', lambda:captured)
    config, attn = donor()
    m = CleanMamba(attn, config)
    with torch.no_grad():
        m.in_weight[-13:].normal_(0, .02)
        m.mimo_o.normal_(0, .3)
    inputs = torch.randn(2, 19, 16)*.2
    reference = m(inputs, reference=True)
    grad_reference = torch.autograd.grad(reference.square().mean(), tuple(m.parameters()))
    components = m.components(inputs)
    output = chunked_scan(*components, m, block=8, checkpoint=checkpoint, compiled=compiled)
    actual = torch.nn.functional.linear(output.flatten(-2), m.out_weight)
    grad_actual = torch.autograd.grad(actual.square().mean(), tuple(m.parameters()))
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
    for a, b in zip(grad_actual, grad_reference):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-3)


def test_frozen_real_qwen_block_mamba_gradients_and_teacher_mask_parity():
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, vocab_size=37)
    cfg._attn_implementation = 'sdpa'
    teacher = Qwen3Model(cfg).eval().requires_grad_(False)
    tokens = torch.tensor([[1, 5, 3, 7, 2]])
    hidden = teacher.embed_tokens(tokens)
    pos = torch.arange(tokens.shape[1])[None]
    rope = teacher.rotary_emb(hidden, pos)
    original = {k:v.clone() for k, v in teacher.state_dict().items()}
    manual = hidden
    for layer in teacher.layers:
        manual = layer(manual, position_ids=pos, position_embeddings=rope)
    torch.testing.assert_close(teacher.norm(manual), teacher(tokens).last_hidden_state)
    m = CleanMamba(teacher.layers[0].self_attn, cfg)
    target = teacher.layers[0](hidden, position_ids=pos, position_embeddings=rope).detach()
    pred = replaced_block(teacher.layers[0], m, hidden, reference=True)
    layer_loss(pred, target).backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(v, original[k]) for k,v in teacher.state_dict().items())
    assert m.in_weight.grad is not None and m.in_weight.grad.abs().sum() > 0


def test_current_resume_keeps_adam_cursor_and_refuses_contract_change(tmp_path):
    cfg, attn = donor()
    bank = torch.nn.ModuleDict({'0': CleanMamba(attn, cfg)})
    opt = torch.optim.AdamW(bank.parameters(), lr=.001)
    bank['0'](torch.randn(1, 3, 16), reference=True).square().mean().backward()
    opt.step(); opt.zero_grad()
    engine = SimpleNamespace(bank=bank, optimizer=opt, step=3, cursor=192,
        effective=16, length=4, contract={'source':'original'})
    save_checkpoint(engine, tmp_path)
    saved = {k:v.clone() for k,v in bank.state_dict().items()}
    with torch.no_grad():
        bank['0'].out_weight.add_(7)
    restore_checkpoint(engine, tmp_path/'CURRENT.pt')
    assert all(torch.equal(v, saved[k]) for k,v in bank.state_dict().items())
    assert len(opt.state) > 0
    engine.contract = {'source':'changed'}
    with pytest.raises(ValueError, match='contract differs'):
        restore_checkpoint(engine, tmp_path/'CURRENT.pt')
    assert sorted(p.name for p in tmp_path.iterdir()) == ['CURRENT.pt']


def test_actual_alignment_update_partition_and_frozen_teacher(monkeypatch):
    from transformers import Qwen3Config, Qwen3Model
    from extent.torch_train_170 import Alignment
    import types
    cfg = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4, vocab_size=37)
    cfg._attn_implementation = 'sdpa'
    teacher = Qwen3Model(cfg).eval().requires_grad_(False)
    original = {k:v.clone() for k,v in teacher.state_dict().items()}
    tokens = torch.randint(0, 37, (3, 5))
    def build():
        e = Alignment.__new__(Alignment)
        e.teacher = teacher
        e.bank = torch.nn.ModuleDict({'0':CleanMamba(teacher.layers[0].self_attn, cfg)})
        def reference(self, inputs, **kw):
            return CleanMamba.forward(self, inputs, reference=True)
        e.bank['0'].forward = types.MethodType(reference, e.bank['0'])
        e.optimizer = torch.optim.AdamW(e.bank.parameters(), lr=.001, weight_decay=0)
        e.timer = SimpleNamespace(phase=lambda name:contextlib.nullcontext(), finish=lambda:{})
        e.batch = lambda cursor,count:tokens[cursor//5:cursor//5+count]
        e.length, e.effective, e.cursor, e.step = 5, 3, 0, 0
        return e
    a, b = build(), build()
    row_a = a.update(1, lr=.001, trace=True)
    row_b = b.update(2, lr=.001, trace=True)
    assert a.step == b.step == 1 and a.cursor == b.cursor == 15
    assert row_a['loss'] == pytest.approx(row_b['loss'], abs=1e-7)
    assert 'mixer_relative_mse' in row_a['layers'][0]
    for key,value in a.bank.state_dict().items():
        torch.testing.assert_close(value, b.bank.state_dict()[key], atol=2e-6, rtol=2e-4)
    assert all(torch.equal(value, original[key]) for key,value in teacher.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())


def test_schedule_warmup_decay_and_import_without_jax():
    assert learning_rate(0) == pytest.approx(.001/46)
    assert learning_rate(45) == pytest.approx(.001)
    assert learning_rate(1525) == pytest.approx(.0001)
    subprocess.run([sys.executable, '-c',
        "import sys; import engine_torch170; import extent.torch_clean_170; import extent.status_autostart; assert 'jax' not in sys.modules"], check=True)


def test_fast_stop_discards_incomplete_eval_before_gpu_work(tmp_path):
    from extent.torch_train_170 import Alignment
    engine = Alignment.__new__(Alignment)
    engine.stop_dir = tmp_path
    (tmp_path/'FAST_STOP').touch()
    assert engine.evaluate() is None


def test_status_shows_alignment_and_full_update_speed_without_legacy_metrics():
    from engine_status import format_report
    report = dict(contract=dict(experiment=170, format='torch-clean-align-v1'), status='running',
        branches={'CLEAN-ALIGN':dict(step=2, source_step=0, accepted_input_tokens=131072,
            history=[dict(step=2, loss=.01, lr=.001, seconds=32, input_tokens_per_second=2048)],
            evaluations=[dict(step=2, teacher_edu_validation_nll=3.1, gqa4_edu_validation_nll=7., gqa7_edu_validation_nll=6.)])})
    text = format_report('clean', report, 0, True, now=1)
    assert '2048 ток/с' in text and 'NLL Edu gqa4: 7.0000' in text
    assert 'Train CE / KL' not in text and 'Скорость ядра:' not in text


def test_remote_cleanup_only_after_verified_sha_and_manifest(tmp_path, monkeypatch):
    import engine_torch170 as launch
    import huggingface_hub
    current = tmp_path/'CURRENT.pt'; current.write_bytes(b'own checkpoint')
    report = tmp_path/'runtime.json'; report.write_text('{"local_checkpoint_step":2}')
    sha = launch.digest_file(current)
    fake = SimpleNamespace(upload_file=lambda **kw: SimpleNamespace(oid='abc'),
        get_paths_info=lambda **kw: [SimpleNamespace(lfs=SimpleNamespace(sha256='wrong'))],
        create_commit=lambda **kw: SimpleNamespace(oid='meta'))
    monkeypatch.setattr(launch, 'remote_api', lambda:fake)
    policy = SimpleNamespace(call=lambda fn, *a, **kw:fn(*a, **kw))
    with pytest.raises(ValueError, match='LFS SHA'):
        launch.sync_remote(tmp_path, report, policy)
    assert current.exists()
    fake.get_paths_info = lambda **kw:[SimpleNamespace(lfs=SimpleNamespace(sha256=sha))]
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda **kw:str(tmp_path/'latest.json'))
    launch.sync_remote(tmp_path, report, policy)
    assert not current.exists()

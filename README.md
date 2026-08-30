# Extent: JAX Mamba-3/MLA hybrid prototype

`Extent-14B` is the model built in this repository. It is a research bring-up
path for a parameter-matched Qwen 14B-class hybrid: 34 Mamba-3 MIMO mixers,
6 MLA mixers, and the Qwen3 SwiGLU MLPs. It targets one TPU v5e-8 slice, but all
reference code and tests run on CPU.

## Private Kaggle or Colab checkout

Use a repository-scoped, read-only SSH deploy key for Kaggle. Do not upload a
personal GitHub SSH key and do not put a token in a clone URL. Generate a
dedicated key locally (leave its passphrase empty because Kaggle is
non-interactive):

```powershell
ssh-keygen -t ed25519 -C "kaggle-extent-readonly" -f "$env:USERPROFILE\.ssh\extent_kaggle"
Get-Content "$env:USERPROFILE\.ssh\extent_kaggle.pub"
```

Add the public line under GitHub repository `Settings -> Deploy keys -> Add
deploy key`, and leave `Allow write access` unchecked. Encode the private key
as one Base64 line and copy it to the clipboard:

```powershell
[Convert]::ToBase64String(
    [IO.File]::ReadAllBytes("$env:USERPROFILE\.ssh\extent_kaggle")
) | Set-Clipboard
```

Store that clipboard value as a secret named `EXTENT_DEPLOY_KEY_B64`. In
Kaggle, use the notebook Secrets settings. In Google Colab, use the key icon in
the left sidebar and enable notebook access for the secret. Never print or
decode the secret into notebook output.
Base64 is transport encoding, not encryption—the Kaggle secret remains the
security boundary. It avoids multiline-secret corruption that otherwise causes
OpenSSH `error in libcrypto`. A deploy key is scoped to one repository and is
read-only by default. In a fresh Kaggle or Colab session, clone with:

```python
from pathlib import Path
import base64
import os
import shlex
import subprocess

try:
    from google.colab import userdata

    private_key_b64 = userdata.get("EXTENT_DEPLOY_KEY_B64")
    notebook_workdir = "/content"
    secret_provider = "google-colab"
except ImportError:
    from kaggle_secrets import UserSecretsClient

    private_key_b64 = UserSecretsClient().get_secret("EXTENT_DEPLOY_KEY_B64")
    notebook_workdir = "/kaggle/working"
    secret_provider = "kaggle"

if not private_key_b64:
    raise RuntimeError(
        f"EXTENT_DEPLOY_KEY_B64 is unavailable from {secret_provider}"
    )
private_key_bytes = base64.b64decode(private_key_b64.strip(), validate=True)
if not private_key_bytes.startswith(b"-----BEGIN OPENSSH PRIVATE KEY-----"):
    raise ValueError("decoded Kaggle secret is not an OpenSSH private key")
if not private_key_bytes.rstrip().endswith(b"-----END OPENSSH PRIVATE KEY-----"):
    raise ValueError("decoded Kaggle secret has a truncated OpenSSH footer")
private_key_bytes = private_key_bytes.rstrip() + b"\n"

ssh_dir = Path.home() / ".ssh"
ssh_dir.mkdir(mode=0o700, exist_ok=True)
private_key = ssh_dir / "extent_kaggle"
known_hosts = ssh_dir / "known_hosts"
private_key.write_bytes(private_key_bytes)
private_key.chmod(0o600)
known_hosts.write_text(
    "github.com ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl\n",
    encoding="utf-8",
)
known_hosts.chmod(0o600)

key_check = subprocess.run(
    ["ssh-keygen", "-y", "-P", "", "-f", str(private_key)],
    text=True,
    capture_output=True,
)
if key_check.returncode != 0:
    raise RuntimeError(
        "decoded deploy key failed local ssh-keygen validation: "
        + key_check.stderr.strip()
    )

git_env = os.environ.copy()
git_env["GIT_SSH_COMMAND"] = (
    f"ssh -i {shlex.quote(str(private_key))} -o IdentitiesOnly=yes "
    f"-o UserKnownHostsFile={shlex.quote(str(known_hosts))} "
    "-o StrictHostKeyChecking=yes"
)
subprocess.run(
    ["git", "clone", "git@github.com:KremlevLev/extent.git"],
    cwd=notebook_workdir,
    env=git_env,
    check=True,
)
```

The pinned `known_hosts` entry is GitHub's published Ed25519 host key. If
GitHub announces a host-key rotation, update it from the official fingerprint
page rather than disabling strict host checking.

## What is implemented

- A reusable Flax `Mamba3MIMO` block with data-dependent decay, the
  exponential-trapezoidal update, complex/rotary state channels, rank-4 MIMO,
  and an FP32 recurrent state.
- The EXP-019-selected RoRoPE+BKV attention contract in all six retained
  attention layers: rank-448 activation-PCA cache plus one 128-element rotary
  key, reducing retained-attention KV elements by 71.875% versus Qwen3 GQA.
- The earlier MaxText-style MLA module remains available as an explicit
  scaffold/ablation, but it is no longer the production Extent-14B attention.
- A pre-norm hybrid decoder and causal LM head.
- BF16 parameters and gradients, with FP32 recurrence/softmax/logits for
  numerical stability.
- Lion with a single BF16 momentum buffer, optional gradient accumulation, full remat,
  and conservative `data/fsdp/tensor = 1/4/2` partitioning.
- Streaming-friendly Qwen mapping plans, SVD helpers for MLA, and an explicitly
  labelled heuristic Q/K/V -> Mamba input-projection transplant.

The `lax.scan` recurrence is the correctness implementation, not the final
throughput kernel. It is JIT/autodiff compatible on TPU. Before long runs,
replace `mamba3_reference_scan` with a chunked Pallas kernel and retain its
public tensor/parameter contract.

## Smoke test

```bash
python -m pytest -q tests/test_research_prototype.py
python -m scripts.smoke_prototype --sequence-length 8
```

The legacy `tests/test_qwen_base.py` is excluded from collection because it
downloads Qwen3-14B and runs work at import time. Treat it as a manual checkpoint
integration script, not an offline unit test.

To verify real multi-device parameter and optimizer sharding on two GPUs:

```bash
python -m scripts.sharded_smoke --sequence-length 8 --steps 1
```

For a two-device `tensor=2` mesh, the tiny model should report approximately
`partitioned_param_arrays=20/47` and
`partitioned_train_state_arrays=40/97`. The second number includes the sharded
Lion momentum. Exact loss can vary slightly between CUDA and TPU.
The step also reports `grads_finite` and `nonfinite_grad_leaves`. Gradient norm
and clipping use FP32 reductions even though stored gradients remain BF16. A
non-finite step is reported and skipped instead of corrupting parameters.
When CUDA reports non-finite gradients, the launcher recompiles a diagnostic
backward and prints the exact parameter paths. On pre-Ampere GPUs, compare with
`--compute-dtype float32`; parameters and stored gradients remain BF16.
The sharded launcher defaults to `--compute-dtype auto`: it selects FP32 compute
on T4/V100-class GPUs and BF16 on TPU v5e or BF16-capable modern GPUs. Passing
`--compute-dtype bfloat16` explicitly keeps the failure-reproduction path.

Do not pass `--config config/hybrid_14b_v5e8.yaml --allow-full-model` on ordinary
two-GPU Kaggle instances: the ideal BF16 weights + gradients + Lion state alone
would require about 41 GiB per GPU. The full configuration is intended for eight
devices and should still begin with a zero/one-step HBM bring-up.

Before reserving v5e-8, run the exact shape-only audit on any machine (including
v5e-1). It traces all parameter shapes but allocates no 14B arrays:

```python
from scripts.full_model_preflight import main as full_preflight
full_preflight([])
```

Also audit the complete transplant ownership and selected mixer contracts. This
allocates neither model weights nor checkpoint tensors:

```python
from scripts.full_hybrid_transplant_preflight import main as transplant_preflight

transplant_preflight(["--require-ready"])
```

The expected production result is `443/443` pinned Qwen source tensors owned,
34 Mamba layers assigned to `INIT-K-balanced-qkvo-lift`, six attention layers
assigned to the EXP-019 RoRoPE+BKV conversion, and `verdict=GO`. The generated
JSON is written under `output/` for audit but is intentionally not committed.

## EXP-061: one long production-aligned TPU run

EXP-061 is the next publication experiment, designed to use one v5e-8 session
for roughly 6–7 hours rather than spending an allocation on one HBM probe. It
trains paired random/exact-lift endpoints at 12 layers drawn only from the final
34-layer Mamba schedule, then evaluates nested 4/8/12-layer compositions with
three paired seeds and end-to-end NLL. It is resumable at every completed layer,
keeps large Qwen shards in RAM when available, writes durable stage/result
JSONs, deletes endpoint payloads only after successful final evaluation, and
sends Telegram start/completion/failure notifications.

Run it inside the already initialized Kaggle notebook process (never through
`!python` or `%run`):

```python
from scripts.qwen_production_aligned_scaling_campaign import main as run_exp061

result = run_exp061([])
```

Download only these small final artifacts:

- `/kaggle/working/output/extent-production-aligned-scaling-campaign-summary.md`
- `/kaggle/working/output/extent-production-aligned-scaling-campaign.json`
- `/kaggle/working/output/exp061-stage-manifest.json`

Do not download activation arrays or endpoint payloads. If a managed Kaggle run
is interrupted, rerun the same cell with the preserved output attached and the
default resume behavior; completed, hash-checked layer boundaries are skipped.

## EXP-062: real full-hybrid materialization bring-up

EXP-062 is the first non-shape-only construction of Extent-14B. It generates
teacher calibration inputs for the six retained attention layers, downloads the
pinned Qwen3-14B checkpoint into RAM-backed storage, creates the global sharded
14.766B parameter tree, imports all preserved tensors, materializes 34 balanced
exact-lift Mamba mixers and six rank-448 RoRoPE+BKV mixers, allocates Lion, and
compiles full-model forward probes at contexts 8/32/128.

Run in the notebook Python process on v5e-8:

```python
from scripts.full_hybrid_materialization_campaign import main as run_exp062

result = run_exp062([])
```

After the forward-only artifact passes, reuse the same live v5e-8 session for
the final bring-up gate. This keeps the BF16 Lion state resident and compiles
one full-model gradient health probe at context 8, but does not update weights:

```python
import importlib
import scripts.full_hybrid_materialization_campaign as exp062

importlib.reload(exp062)
result = exp062.main([
    "--probe-contexts", "8",
    "--backward-probe-context", "8",
])
```

The small result is
`/kaggle/working/output/extent-full-hybrid-backward-bringup.json`. A PASS
requires finite loss, finite global BF16 gradient norm, finite maximum absolute
gradient, and zero non-finite gradient leaves. This is a numerical/HBM gate,
not a training-quality measurement.

The campaign writes a partial JSON after every materialized layer, sends
Telegram start/completion/failure messages, and writes the final small artifact
to `/kaggle/working/output/extent-full-hybrid-materialization.json`. Download
only that JSON (or `exp062-materialization-failure.json` on failure). Calibration
NPY files and the 27.5 GiB source checkpoint are regenerable and must not be
downloaded.

This first bring-up does not serialize the resulting 27.5 GiB Extent parameter
tree. A PASS is required before adding sharded safetensors export/Hugging Face
upload; otherwise a large but invalid checkpoint could be published.

## EXP-063: Mamba-3 in the Qwen homotopy screen

This is the first controlled test of the proposed **M3Q** recovery method on
the immutable `Qwen/Qwen3-1.7B-Base` checkpoint. It compares random, flat-QKVO,
and balanced exact-lift controls with three schedules that gradually replace
teacher attention by the exact-lift Mamba-3 block inside the frozen decoder.
Five registered boundary/interior layers, three paired seeds, 8,192 recovery
steps, and six checkpoints measure both final error and recovery-curve AUC.
Every held-out checkpoint is evaluated with the teacher bridge removed.

Run only this final cell after the normal Kaggle setup cells:

```python
from scripts.m3q_homotopy_campaign import main as run_exp063

result = run_exp063([])
```

Download only:

- `/kaggle/working/output/extent-m3q-homotopy-screen-summary.md`
- `/kaggle/working/output/extent-m3q-homotopy-screen.json`
- `/kaggle/working/output/exp063-failure.json` only if Telegram reports failure

Activation arrays and the source checkpoint are regenerated and must not be
downloaded.

Only on v5e-8, after reviewing the report, initialize parameter shards with:

```python
full_preflight(["--initialize-params"])
```

The guarded initialization refuses to run unless the actual eight-device mesh
matches `data=1, fsdp=4, tensor=2`. It creates weights only—not Lion state or a
full training step.

After parameter initialization passes, restart with a fresh TPU runtime and
probe the real persistent allocation for weights plus Lion momentum:

```python
from scripts.full_model_preflight import main as full_preflight
full_preflight(["--initialize-optimizer"])
```

This still does not run forward/backward and does not allocate gradients or
activations. It prints parameter, optimizer, and combined bytes for every TPU.
Run it in the notebook kernel; do not use `!python`, `%run`, or a subprocess
after JAX has initialized libtpu.

On a fresh Kaggle TPU runtime:

```bash
pip install -r requirements-tpu.txt
python -m scripts.smoke_prototype --sequence-length 8
```

In Google Colab, libtpu is exclusive to one Python process. After importing JAX
in the notebook kernel, do not launch `!python` or `!pytest`. Run validation in
that same process instead:

```python
import pytest
pytest.main(["-q"])

from scripts.sharded_smoke import main as sharded_smoke
sharded_smoke(["--sequence-length", "8", "--steps", "1"])
```

## Kaggle Telegram TPU notification

Create a bot with Telegram's `@BotFather`, open the bot, press **Start**, and
send it any message. Store the token in Kaggle **Add-ons > Secrets** as
`TELEGRAM_BOT_TOKEN`; never paste it into a notebook or commit it.

To discover the destination chat id, enable notebook access for that secret and
run in the notebook process:

```python
from scripts.telegram_tpu_notifier import main as telegram_tpu_notifier
telegram_tpu_notifier(["--show-chat-ids"])
```

Store the printed id as a second Kaggle secret named `TELEGRAM_CHAT_ID`. On the
next fresh TPU session, after installing requirements, run:

```python
from scripts.telegram_tpu_notifier import main as telegram_tpu_notifier

NOTIFIER_ENABLED = True
telegram_tpu_notifier(
    ["--message", "experiment runtime initialized"],
    enabled=NOTIFIER_ENABLED,
)
```

The notifier reports whichever runtime JAX actually sees: TPU, GPU, or CPU. If
JAX initialization itself fails, it attempts to send that failure too. Set
`NOTIFIER_ENABLED = False` to disable all checks and network calls. It cannot
notify before Kaggle starts the notebook kernel, because no project code is
running before that point. Do not launch it through `!python` or `%run`;
libtpu must stay in the notebook process.

Start real bring-up with `config/hybrid_14b_v5e8.yaml`. Its sequence length is
deliberately 1024 and micro-batch size is one. Compile a single step and inspect
HBM before raising sequence length. The analytical model size is approximately
14.7B parameters; BF16 weights + BF16 gradients + one BF16 Lion moment consume
about 10.3 GiB per device under ideal eight-way sharding, before activations,
XLA temporaries, executable buffers, and unevenly replicated parameters.
`gradient_accumulation_steps` starts at one: Optax accumulation would add a
further full BF16 buffer (roughly 3.4 GiB/device for this model).

## MaxText boundary

`config/maxtext_mla_v5e8.yml` and `extent.maxtext_adapter` expose the
upstream MaxText MLA configuration surface. Upstream MaxText does not currently
provide a Mamba-3 decoder block, so `extent.model` owns layer scheduling.
This avoids claiming a drop-in MaxText model that cannot compile; the MLA names
are kept compatible so its optimized kernel can replace the reference attention
without another checkpoint conversion.

## Pinned Qwen3-14B source

The source model is pinned to `Qwen/Qwen3-14B` revision
`40c069824f4251a91eefaf281ebe4c544efd3e18`. Validate its official config,
safetensors index, the complete 443-tensor JAX teacher mapping, and all 203
direct student mappings without downloading weights:

```python
from scripts.qwen_checkpoint_preflight import main as qwen_preflight
qwen_preflight([])
```

Expected output includes:

```text
teacher_mapping=PASS tensors=443 parameters=14,768,307,200 coverage=100%
direct_mapping=PASS tensors=203 parameters=12,251,714,560
```

The teacher result proves a bijective checkpoint-to-JAX parameter-tree contract;
it is not yet numerical logits parity. The direct student map covers
12,251,714,560 parameters (83.30% of the final target):
embeddings, output head, final norm, and every decoder MLP/norm. Attention
projections and Q/K norms are deliberately left for controlled MLA and Mamba-3
transplant experiments.

Only on a machine with at least 35 GiB of free disk, download the eight pinned
weight shards:

```python
from scripts.download_qwen_checkpoint import main as download_qwen
download_qwen(["--output-dir", "/path/with/enough/space/qwen3-14b"])

qwen_preflight(["--model-dir", "/path/with/enough/space/qwen3-14b"])
```

The loader reads one tensor at a time and constructs global JAX arrays directly
in the target sharding. Do not create the Lion state before importing weights.
On v5e-8, the guarded full initializer/import path is:

```python
from scripts.full_model_preflight import main as full_preflight
full_preflight([
    "--initialize-params",
    "--qwen-model-dir", "/path/with/enough/space/qwen3-14b",
])
```

This imports only the exactly preserved 12.252B parameters. It does not claim
teacher parity yet and does not initialize MLA/Mamba-3 from GQA projections.

## Qwen3 decoder-layer numerical parity

Before any mixer conversion, compare a real source layer against the official
PyTorch implementation. This downloads only the shard containing layer 0
(`model-00001-of-00008.safetensors`, approximately 3.58 GiB), runs both sides
in FP32, and applies frozen error gates:

```python
%cd /kaggle/working/extent
!pip install -q -r requirements-parity.txt

from scripts.qwen_layer_parity import main as qwen_layer_parity
qwen_layer_parity([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--sequence-length", "4",
    "--result-json", "/kaggle/working/qwen3-layer-parity-result.json",
])
```

The script requires `transformers==4.51.0`, the version recorded by the pinned
Qwen3 config. It prints `PARITY-PASS` only when both maximum absolute error and
relative L2 error pass their predeclared thresholds. Run this on a CPU or GPU
notebook; TPU is not required.

After real layer parity passes, measure the immediate GQA-to-MLA shock using the
same downloaded shard:

```python
from scripts.qwen_mla_shock import main as qwen_mla_shock
qwen_mla_shock([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--sequence-length", "4",
    "--kv-rank", "512",
    "--rope-dim", "64",
    "--result-json", "/kaggle/working/qwen3-mla-shock-layer0.json",
])
```

This compares random MLA, copied Q/output projections with random KV, joint-SVD
with all eight source RoPE-key heads, and the more aggressive one-RoPE-head
target. It reports clean mixer-output error separately from residual-masked
decoder-output error. These are initialization diagnostics, not recovered-model
quality results.

Use the measured layer-0 artifact to isolate the KV-rank bottleneck. The sweep
computes one rank-1536 factorization and reuses its leading factors for every
lower rank, so the comparison is nested and deterministic:

```python
from scripts.qwen_mla_rank_sweep import main as qwen_mla_rank_sweep
qwen_mla_rank_sweep([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--sequence-length", "4",
    "--rope-dim", "64",
    "--ranks", "256,512,1024,1536",
    "--result-json", "/kaggle/working/qwen3-mla-rank-sweep-layer0.json",
])
```

Rank 1536 is the no-low-rank-loss control for the 64-dimensional partial-RoPE
layout. The grouped variant has no cache reduction at that rank; its purpose is
to isolate partial-RoPE damage rather than serve as a deployable configuration.

Finally, probe whether the short-sequence partial-RoPE result survives increasing
positions. This is a controlled synthetic-prefix diagnostic, not a long-context
quality benchmark:

```python
from scripts.qwen_mla_context_sweep import main as qwen_mla_context_sweep
qwen_mla_context_sweep([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--lengths", "4,64,256,1024",
    "--compressed-rank", "512",
    "--rope-dim", "64",
    "--tail-tokens", "128",
    "--result-json", "/kaggle/working/qwen3-mla-context-sweep-layer0.json",
])
```

The script reports error over all tokens and separately over the final 128
tokens. It compares rank 512 against the no-SVD-loss rank 1536 control with both
grouped and shared RoPE keys.

After recording the context-length result, compare partial-RoPE widths under the
same deployable shared-cache budget. The latent rank is reduced as RoPE width
grows, so every `shared_fixed_cache` variant stores exactly 576 elements per
token per layer:

```python
from scripts.qwen_mla_rope_width_sweep import main as qwen_mla_rope_width_sweep
qwen_mla_rope_width_sweep([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--sequence-length", "1024",
    "--rope-widths", "32,64,96,128",
    "--target-cache-elements", "576",
    "--tail-tokens", "128",
    "--result-json", "/kaggle/working/qwen3-mla-rope-width-sweep-layer0.json",
])
```

The resulting latent ranks are 544, 512, 480, and 448. For every width the
script also runs grouped- and shared-RoPE controls at the maximum joint rank,
separating positional-frequency removal and RoPE-head aggregation from the
fixed-budget low-rank tradeoff.

The naive shared-head sweep is followed by a standard TransMLA RoRoPE baseline.
This fits per-frequency head rotations on one calibration sequence and evaluates
on an independent sequence. It isolates positional decoupling before FreqFold or
joint KV compression:

```python
from scripts.qwen_rorope_shock import main as qwen_rorope_shock
qwen_rorope_shock([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--calibration-length", "1024",
    "--sequence-length", "1024",
    "--rope-components", "1,2,4,8",
    "--tail-tokens", "128",
    "--calibration-seed", "123",
    "--evaluation-seed", "124",
    "--result-json", "/kaggle/working/qwen3-rorope-shock-layer0.json",
])
```

The eight-component result is an implementation invariant and should be nearly
identical to the source mixer. The one-component result measures the shock of
the single shared RoPE-head target after PCA concentration, rather than naive
head averaging. This launcher is a GPU diagnostic; TPU is not required.

Next, keep that one-head RoPE cache fixed at 128 elements while grouping nearby
frequencies for joint PCA. This is the controlled FreqFold-style positional
diagnostic before balanced or covariance-aware KV compression:

```python
from scripts.qwen_freqfold_sweep import main as qwen_freqfold_sweep
qwen_freqfold_sweep([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--calibration-length", "1024",
    "--sequence-length", "1024",
    "--folds", "1,2,4,8",
    "--tail-tokens", "128",
    "--calibration-seed", "123",
    "--evaluation-seed", "124",
    "--result-json", "/kaggle/working/qwen3-freqfold-sweep-layer0.json",
])
```

Fold 1 must reproduce the EXP-015 one-component metrics. Larger folds trade an
approximation between neighboring RoPE frequencies for a richer PCA subspace,
without changing the retained RoPE-cache width. TPU is not required.

After selecting the positional transform, compress its seven NoPE key
components jointly with the original values. The total target cache is fixed at
576 elements: 128 RoPE plus a rank-448 latent. This probe compares plain
activation PCA with TransMLA-style BKV balancing:

```python
from scripts.qwen_balanced_kv_probe import main as qwen_balanced_kv_probe
qwen_balanced_kv_probe([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--layer-index", "0",
    "--calibration-length", "1024",
    "--sequence-length", "1024",
    "--latent-rank", "448",
    "--tail-tokens", "128",
    "--calibration-seed", "123",
    "--evaluation-seed", "124",
    "--svd-seed", "0",
    "--result-json", "/kaggle/working/qwen3-balanced-kv-probe-layer0.json",
])
```

This remains a GPU layer diagnostic. It reports an uncompressed RoRoPE control,
plain rank-448 activation PCA, and rank-448 BKV-balanced activation PCA. The
activation basis is not yet a deployable checkpoint mapping.

Replace the under-sampled Gaussian calibration with a pinned real-text token
prefix and actual Qwen3 embedding rows. Install the small calibration-only
dependency set once, then run the same cache-matched comparison:

```python
%cd /kaggle/working/extent
!pip install -q -r requirements-calibration.txt

from scripts.qwen_real_text_kv_probe import main as qwen_real_text_kv_probe
qwen_real_text_kv_probe([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--dataset-cache-dir", "/kaggle/working/extent-calibration-cache",
    "--layer-index", "0",
    "--calibration-tokens", "8192",
    "--sequence-length", "1024",
    "--latent-rank", "448",
    "--tail-tokens", "128",
    "--svd-seed", "0",
    "--result-json", "/kaggle/working/qwen3-real-text-kv-probe-layer0.json",
])
```

The pinned WikiText-2 train prefix is used only for calibration diagnostics.
The first 8192 packed tokens fit the PCA basis and the following 1024 tokens are
held out. Randomized PCA executes on the active JAX GPU; TPU is not required.

Validate the frozen RoRoPE-BKV recipe through the normal Flax module and its
explicit compressed-cache interface:

```python
%cd /kaggle/working/extent
!pip install -q -r requirements-calibration.txt

from scripts.qwen_deployable_mla_parity import main as qwen_deployable_mla_parity
qwen_deployable_mla_parity([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--dataset-cache-dir", "/kaggle/working/extent-calibration-cache",
    "--layer-index", "0",
    "--calibration-tokens", "8192",
    "--sequence-length", "1024",
    "--latent-rank", "448",
    "--tail-tokens", "128",
    "--svd-seed", "0",
    "--result-json", "/kaggle/working/qwen3-deployable-mla-parity-layer0.json",
])
```

Run this on a Kaggle GPU, not TPU. Success ends with `DEPLOYABLE-MLA-PASS`.
Download the result JSON for the experiment archive. This is a correctness
reference with an actual 448-element latent cache and 128-element positional
cache; it is not yet an absorbed or optimized incremental-decode kernel.

## Research caveats

The Qwen -> Mamba map is an initialization hypothesis, not exact functional
equivalence. Compare it against random mixer initialization, copied MLP/norms,
and MLA-only conversion under the same recovery-token budget.

The readable JAX recurrence is checked against an independent eager PyTorch
implementation of the official Mamba-3 MIMO one-token equations pinned to
`state-spaces/mamba@e9594ce1c732d97440f0332fdc43170a2294dbfa`. In a notebook,
run the focused cross-framework fixture without starting a second JAX process:

```python
import pytest

exit_code = pytest.main(["-q", "tests/test_mamba3_reference_parity.py"])
print("pytest exit code:", exit_code)
```

Expected: `2 passed` and exit code `0`. This verifies the FP32 mathematical
reference and official parameter shapes. Numerical parity with the optimized
TileLang/CuTe kernel, long-context stability, and production throughput remain
separate accelerator experiments.

Run the first controlled Qwen3-to-Mamba-3 initialization-shock experiment on a
Kaggle GPU. It compares one shared random base, output-only transfer, the
Mamba-in-the-Llama Q/K/V/O port, copied SISO information, and distinct MIMO
channel allocation:

```python
%cd /kaggle/working/extent
!pip install -q -r requirements-calibration.txt

from scripts.qwen_mamba3_shock import main as qwen_mamba3_shock
qwen_mamba3_shock([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--dataset-cache-dir", "/kaggle/working/extent-calibration-cache",
    "--layer-index", "0",
    "--sequence-length", "128",
    "--seed", "123",
    "--result-json", "/kaggle/working/qwen3-mamba3-shock-layer0.json",
])
```

Use T4/P100 rather than TPU for this diagnostic. Success ends with
`MAMBA3-SHOCK-PASS`; download the JSON even when a transplanted variant is worse
than random, because negative results determine which recovery ablations are
scientifically justified.

If the direct transplant has negligible mixer cosine, test whether its frozen
recurrent features can support a teacher-aligned linear readout. Ridge selection
uses only an internal calibration split; the final 128 tokens remain held out:

```python
%cd /kaggle/working/extent
!pip install -q -r requirements-calibration.txt

from scripts.qwen_mamba3_readout_probe import main as qwen_mamba3_readout_probe
qwen_mamba3_readout_probe([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--dataset-cache-dir", "/kaggle/working/extent-calibration-cache",
    "--layer-index", "0",
    "--calibration-tokens", "256",
    "--selection-tokens", "64",
    "--evaluation-tokens", "128",
    "--ridge-values", "1e-2,1e-3,1e-4",
    "--seed", "123",
    "--result-json", "/kaggle/working/qwen3-mamba3-readout-probe-layer0.json",
])
```

Use a Kaggle GPU. Success ends with `MAMBA3-READOUT-PROBE-PASS`; this means the
linear solves and finite checks succeeded, not that frozen Mamba features passed
a quality threshold. Download the JSON so held-out gains and failures are both
retained.

The same launcher now also performs EXP-023: after fitting the raw-hidden
control, it fits each frozen Mamba representation only to the remaining teacher
residual. Re-run it after pulling the latest commit and use a new artifact name:

```python
from scripts.qwen_mamba3_readout_probe import main as qwen_mamba3_readout_probe
qwen_mamba3_readout_probe([
    "--cache-dir", "/kaggle/working/qwen3-layer-parity",
    "--dataset-cache-dir", "/kaggle/working/extent-calibration-cache",
    "--layer-index", "0",
    "--calibration-tokens", "256",
    "--selection-tokens", "64",
    "--evaluation-tokens", "128",
    "--ridge-values", "1e-2,1e-3,1e-4",
    "--seed", "123",
    "--result-json", "/kaggle/working/qwen3-mamba3-context-probe-layer0.json",
])
```

Compare each `raw_plus_mamba_heldout` entry with
`raw_normalized_hidden_control.heldout`. The Mamba representation adds useful
context only when the former improves held-out relative L2 and cosine.

## Time-bounded attention bridge screen

When only four TPU hours are available, defer the 7--8-hour EXP-053 campaign and
run EXP-054 instead. It compares canonical random initialization, the rejected
direct QKVO control, an Apple-inspired learned linear-attention bridge, a
MOHAWK-inspired Mamba-3 matrix-orientation warm start, and their composition at
Qwen layers 18 and 0. Every arm receives the same decoder-aware recovery budget;
initializer construction work is reported separately.

```python
from scripts.qwen_bridge_ablation_campaign import main as run_bridge_screen

result = run_bridge_screen(["--max-wall-hours", "3.5"])
print(result["status"], result["aggregate"]["screening_gate_passed"])
print("/kaggle/working/output/extent-bridge-ablation-campaign-summary.md")
```

Run this as the first JAX use in a fresh TPU v5e-8 notebook process. The campaign
writes after each completed arm, sends Telegram start/final/failure messages,
and returns `deadline_partial` rather than discarding finished comparisons when
the internal wall deadline is reached. Download the compact summary, not the
activation arrays.

EXP-054 exposed unstable full-feature RoPE normalization and did not pass. Its
separately numbered correction is EXP-055:

```python
from scripts.qwen_stabilized_bridge_campaign import main as run_stabilized_bridge

result = run_stabilized_bridge([])
print(result["status"], result["aggregate"]["screening_gate_passed"])
```

EXP-055 locks partial RoPE to the canonical Mamba-3 state fraction, restores the
paper's cosine-only bridge objective, and compares all arms for 4,096 matched
recovery steps. It does not overwrite or reinterpret EXP-054.

EXP-056 tests an operator-preserving QKVO SISO-to-MIMO lift. Single-channel and
balanced-rank initializations are functionally equal before training, allowing
their paired recovery trajectories to isolate MIMO optimization geometry. The
resilient v5e-8 entry point is `scripts.qwen_mimo_lift_campaign`.

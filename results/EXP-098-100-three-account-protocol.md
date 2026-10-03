# EXP-098--100: three independent eight-hour Kaggle recovery sessions

Prepared 2026-10-03 after the completed EXP-097 result, before any new outcomes.
User-authorized scope: three TPU v5e-8 accounts, one eight-hour session each.
These are executable independent campaigns, not three successive dependencies.

## Shared registered contract

- Source: `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
  Hash-verified EXP-072-v2 ONPOLICY endpoints at source seeds123/456, 24 Mamba
  and four unchanged GQA layers `[6,13,20,27]`. No MLA or new initializer.
- Restart from unchanged ONPOLICY plus zero-change corrections. EXP-097 trained
  adapters/Adam are preserved as evidence but are not these matched starting states.
- Base BF16 weights frozen; FP32 low-rank A/B coordinates and Adam moments.
  Mamba input/output kernels only, LR3e-4, no decay, batch1, context256,
  max-scaled global clipping1.0. Corrections fold into kernels; actual export
  and inference parity remain separate verification tasks.
- Protected arms exclude in_proj dt/raw_a/trap/angle columns. Their activations
  can still change due to upstream input changes; dynamics are not fully fixed.
- Same ordered training prefix across campaigns, beginning at WikiText103 train
  token33,554,432. EXP098 uses16,384 unique windows; EXP09912,288; EXP1008,192.
  No modulo repeat occurs before the registered endpoint.
- WikiText dataset pin: `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`,
  configuration `wikitext-103-raw-v1`. Validation offset81,920 /32 windows;
  locked test offset262,144 /64 windows, context256.
- Secondary locked cross-domain evaluation: PG-19 **test** offset262,144 /64
  windows using `deepmind/pg19@4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b`
  sorted split manifest and Google-hosted assets. This starts after EXP088's
  complete reserved four-replication PG19 test range ending at262,144.
- New packed ranges relative to registered ranges in the repository; not a
  universal claim of document/semantic disjointness or absence of pretraining
  contamination. The same two model seeds are reused, not new initializations.
  Actual token hashes are checked on every resume; data-capacity errors occur
  before model allocation. Original Qwen is evaluated on both locked tests.
  Local real-tokenizer verification passed all ranges, including the full
  4,194,304-token EXP098 train slice; expected hashes are enforced on the very
  first Kaggle run as well. Auditable range/hash record:
  `results/EXP-098-100-data-preflight.json`. No model weights were downloaded.
- Start and fixed final endpoint are evaluated on both tests, with per-window
  NLL retained. Validation every1,024 is diagnostic; it selects no endpoint,
  LR, arm, or alpha. Do not change gates after seeing outcomes.
- Primary gates are per source seed and require all registered arms complete,
  primary final Wiki test NLL <= control-0.1 and below its unchanged start.
  EXP100 additionally requires its CE control below its start at both seeds.
- Secondary cross-domain gate requires the primary gate plus primary PG19
  final NLL <= control-0.1 and below its PG19 start at both seeds. A Wiki-only
  win is reported explicitly as such, not as robust capability recovery.
- Shared rank32 CE controls are consistency checks, not independent evidence.
  Their paths use identical prefixes, but contracts/horizons and saved states
  remain independent. Equal tokens are not equal parameters or FLOPs.

## Campaigns and budgets

| Account | Experiment | Arms / seeds | Steps per trajectory | Total steps | Training-token presentations |
|---|---|---|---:|---:|---:|
| 1 | EXP098 dynamics protection | ordinary32/protected32 x2 |16,384|65,536|16,777,216|
| 2 | EXP099 protected capacity | protected16/32/64 x2 |12,288|73,728|18,874,368|
| 3 | EXP100 weak anchoring | CE/Qwen-anchor/start-anchor x2 |8,192|49,152|12,582,912|

EXP097 completed83,163 steps in7.329166h: campaign average0.317268s/completed
step, including startup, JIT, evaluation, failed attempts and saves. This is not
a measured steady-state step benchmark or a FLOP-normalized comparison.
Naively scaled totals are5.78h/6.50h/4.33h. Rank64 changes cost; EXP100 adds
teacher/anchor forward passes, so its4.33h estimate excludes that added cost.
One-session completion is not guaranteed.

Each new invocation defaults to8.0h **including**45min finalization reserve.
The training deadline is7.25h after runner entry, counting its source/data/model
startup. Earlier notebook setup is outside this timer. Longer budget overrides
are rejected. There is no artificial wait to consume unused TPU time.
Fixed horizons are rotated across all arms/seeds. A partial run preserves the
same final endpoint and resumes next session; it is not shortened into a pass.
Seed123 uses listed arm order; seed456 reverses arm order to counterbalance
startup/JIT timing. Layer order and token order are unchanged.

## Durability and health

- Local atomic adapter+Adam binary saves every256 completed steps, at horizon
  boundaries, first norm-only event **before** its update, failure and budget stop.
- Bundled Hub commits at most once/10min per account, plus terminal save. Across
  three accounts this is ordinarily about18 periodic commits/hour, excluding
  terminal/other jobs; no per-metric commits. 429 cools down1h and local training
  continues; other transient failures defer120s; auth401/403 fails explicitly.
- Checkpoints restore optimizer count/moments, coordinates, data cursor and exact
  source/config/data/implementation hashes. Missing state never restarts silently.
- Each campaign has independent RAM/Qwen-cache/checkpoint/HF namespaces. No big
  checkpoint is transferred through the user's computer. Source and old result
  artifacts are not deleted. External VM kill can lose the unsynced tail.
- Diagnostics separate forward/target/loss/gradient elements/naive norm/current
  coordinates/current Adam/proposal. Raw norm overflow alone does not reject a
  finite SAFE update. No nan_to_num, skipped batches or optimizer reset.
- Latest health, maximum observed log10 gradient norm, forward/gradient failure
  count and attempted-training-step seconds are recorded. Timing includes JIT
  and excludes host diagnostics/evaluation/saving; it is not pure FLOP timing.
- Numerical failure affects one arm and retains last finite state. Structural
  errors raise with stage/traceback. All-terminal failed campaigns rerun without
  model allocation. A final-training checkpoint lacking test evaluation resumes
  evaluation without repeating completed updates.
- Telegram start/end/budget-partial/caught-error callbacks use existing secrets.
  External kill cannot guarantee a callback.

## Notebook use

Existing clone/install/secrets/setup cells stay above. The checkout must contain
these new modules on main. Run **one** corresponding final cell per account in
the notebook kernel; no `!python`, JAX subprocess or separate pytest process
after TPU initialization. Use Save Version normally. If pip replaces initialized
JAX, restart the kernel before running.

Account1:
```python
from scripts.m3q_dynamics_protection_campaign import main as run_exp098
result = run_exp098([])
```
Account2:
```python
from scripts.m3q_protected_capacity_campaign import main as run_exp099
result = run_exp099([])
```
Account3:
```python
from scripts.m3q_weak_anchor_campaign import main as run_exp100
result = run_exp100([])
```

Send the matching `extent-m3q-*-summary.md` **and** `.json` from
`/kaggle/working/output/` or HF latest files. Stems:
`extent-m3q-dynamics-protection`, `extent-m3q-protected-capacity`,
`extent-m3q-weak-anchor`. HF prefixes match `experiments/exp098-dynamics-protection`,
`experiments/exp099-protected-capacity`, `experiments/exp100-weak-anchor`.
Same final cell resumes partial compatible states. Plan-only is offline:
`run_exp098(["--plan-only"])`, analogously099/100. Do not change scientific
settings between sessions or relabel a numerical survivor as quality recovery.

## Interpretation and next decision

EXP098 asks protection at equal rank and longer horizon; EXP099 asks capacity;
EXP100 asks CE-first weak anchoring. Specific primary arms were fixed before
outcomes. The third arm of099/100 remains secondary, even if it wins afterward.
Rank32 protection was selected on reused EXP097 evaluation; fresh-range tests
here are a next confirmation step, not an independent model-seed replication.
No outcome establishes superiority to random under equal **total** compute,
full recovery to Qwen, long-context quality, MLA, inference speed or14B quality.
Include EXP069/072 warm-start preparation and teacher forwards in future
method-vs-random efficiency comparisons. Preserve positive and negative outcomes.

## Local verification boundary

Real JAX0.6.2 CPU compiled updates and exact binary Adam next-step resume were
checked for all eight registered arm entries. The same eight updates also passed
on an eight-virtual-CPU1/4/2 mesh. Real pinned WikiText103/PG19/tokenizer ranges
and token hashes were checked without model weights. Mocked Hub/TPU workflows
check completed rerun, interrupted training/initial and final evaluation, local binary saves,
429 deferral, auth failure, numerical branch failure and budget/data guards.
This is local correctness evidence, not real TPU HBM/throughput or method outcomes.

Verification: initial focused suite 46 passed /3 skipped; eight virtual-device
compiled arm cases 8 passed; expanded workflow/noncompiled suite 18 passed;
initial/final evaluation interruption regression 2 passed. All runs local CPU.

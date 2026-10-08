# EXP106/107 preparation checks — 8 October2026

ENGINEERING ONLY. No TPU, production Qwen weight allocation, live HF write or
Telegram delivery performed locally.

- Real pinned tokenizer/datasets from local cached artifacts, offline mode:
  registered32768 training windows and all evaluation windows fit; exact SHA,
  dtype and shape in EXP-106-107-data-preflight.json.
- Both entry-point --plan-only calls passed. Source is fixed098; scientific
  settings matched except experiment ID and registered clipping fields.
- CPU with eight virtual devices:42 tests passed in181.02s across
  test_layer_clip_recovery.py, test_paper_decoder_recovery.py and
  test_stable_gradient_clip.py. These include legacy104 fingerprint/contracts.
- Final numerical-failure stop refinement rechecked:4 deadline/failure/cloud
  fresh-runtime-resume cases passed in5.14s. Nonfinite proposals are rejected,
  last safe optimizer cursor retained, campaign stops instead of spending quota
  on remaining seeds after a decisive numerical failure.
- Compiled tiny full-decoder steps for both clips, moment sharding, frozen
  vocabulary and bitwise next-step equality after binary resume.
- Extreme finite gradients1e30 do not overflow clipping; layer-isolation case
  preserves another group's small gradients while global clip suppresses them.
  Zero weight floor and visible NaN handling verified. This synthetic mechanism
  test does not establish the mechanism on the production hybrid.
- Mock complete remote binary roundtrip; manifest/sample readback; corrupted
  sample rejection;401 abort/429 cooldown; transient retry with retained upload
  receipts; destroyed local runtime rebuilt from mock-cloud optimizer state.
- Long schedule remains near peak at2048 and reaches final LR at32768.
  Cross-account aggregator rejects mismatched contracts/window counts/nonfinite
  metrics and does not mark unmatched partial endpoints as primary success.

Production parameter counts/optimizer sizes reuse measured104 scope. The prior
CPU abstract12.32GB/device policy estimate is not a new compiled107 footprint.
New runner measures real compiler/allocator capacity and update/upload durations
on TPU and refuses unsupported capacity. New layer diagnostics and large-state
HF readback can affect throughput; one-session completion is not guaranteed.

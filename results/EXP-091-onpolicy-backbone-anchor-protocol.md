# EXP-091 — on-policy warm start with an anchored Qwen backbone

Status: code prepared; no TPU result yet.

## Question

EXP-072 established that training Mamba blocks on inputs from the assembled
hybrid beats clean-Qwen-only layer training, but the 24-block model remained
far from Qwen. EXP-073/074/075 found that short whole-model recovery from
these endpoints was unstable, especially when copied Qwen weights moved.
Does reducing the *actual Lion update* of copied Qwen weights to 3% while
continuing full Mamba updates improve longer whole-model recovery?

## Matched design

- Pinned Qwen3-1.7B-Base, 24 Mamba-3 blocks and four retained GQA blocks.
- Both arms start from the same hash-verified EXP-072-v2 `ONPOLICY` endpoints
  at seeds 123 and 456. The exact same Qwen parameters fill the remaining
  subtrees. No new local block calibration is performed.
- `ONPOLICY-PLAIN`: ordinary all-parameter BF16 Lion.
- `ONPOLICY-ANCHOR` (primary): identical optimizer and gradients, but the
  *post-Lion updates* of all non-Mamba weights are multiplied by 0.03. The
  Mamba updates and Lion momentum are unchanged. Scaling gradients before
  Lion would not implement this intervention because Lion is sign-based.
- Both arms receive 24,576 updates, context 256, identical train token windows,
  teacher KL plus 0.1 true-token CE, and the same evaluation windows.
  This is 6,291,456 token presentations per trajectory, 25,165,824 total
  over two arms and two seeds. The long-horizon EXP-089 RANDOM curve is only
  contextual: its warm-start compute is unequal and it is not the gate control.
- Primary gate: at **both seeds**, anchor beats plain at step 24,576 in
  held-out excess NLL and prediction KL. Curves at 0/3,072/8,192/16,384/24,576
  are diagnostic. A gate pass would support *longer protected recovery from
  on-policy starts*, not recovery to Qwen or superiority over random init.

## Operations and failure boundaries

One call targets a maximum 8.35-hour Kaggle v5e-8 session and reserves 35
minutes for checkpoint upload/finalization. Actual runtime is unmeasured;
it may finish only part of the four trajectories. A rerun resumes exact model
and Lion state without repeating durable steps. The source manifests are
checked on HF before loading model shards, so missing EXP-069/072 endpoints
fail early. State checkpoints upload at most every 8,192 steps and at the
deadline; JSON and Markdown summaries are published separately. Start,
failure and terminal/partial notifications use the existing Telegram hook.
No local checkpoint download is required of the user.

The 0.03 multiplier is an exploratory intervention, not a selected optimum.
With BF16 weights, small updates can round away; a negative result must not
be read as a universal refutation of backbone protection. Per-run step-0 and
checkpoint metrics, endpoint hashes and data hashes are retained for audit.

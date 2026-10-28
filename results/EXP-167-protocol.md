# EXP-167: matched-token full-decoder learning-rate comparison

Prepared 28 October 2026. No GPU result yet.

Question: does the current full-decoder learning rate limit recovery, or does
raising it merely increase instability or trade one diagnostic domain for another?
EXP166 improved Edu E by0.09290 and C4 D by0.04364 over48,046,080 accepted input
tokens. That observation alone does not identify an LR bottleneck.

All three arms fork **EXP166 CURRENT181145**, not BEST178165. Carry the exact
full-decoder coordinates, all Adam moments/counts26293, and Edu cursor204673.
The166 parent pointer/report/contract, scientific source hashes, external manifest
SHA, optimizer counts, data cursor and A-E NLL replay must agree before training.
Frozen embeddings and final normalization/head remain frozen; trainable decoder
coordinates1,364,173,952. Model remains Qwen3-1.7B-derived hybrid, not14B.

| Arm | Mamba/mixer and MLP LR | Block normalization/pregate LR |
|---|---:|---:|
| x1 control | 1e-5 | 1e-6 |
| x3 | 3e-5 | 3e-6 |
| x10 | 1e-4 | 1e-5 |

All LR groups scale together. Adam beta1.9,beta2.999,epsilon1e-8, stable global
clip1, and the original512-update warmup schedule are unchanged. Carried count
26293 means warmup already finished: no fresh Adam or artificial new warmup.
Loss remains.5CE+.5forwardKL(T1), frozen original Qwen teacher.

Each arm sees the SAME4096 accepted updates x physical batch8 x256 tokens:
**8,388,608 input tokens /8,355,840 next-token targets**. Start window204673,
end237441; final global185241 and Adam30389. No repeat, no cursor skipping,
no teacher-input token double-counting. Across all three arms25,165,824 accepted
input-token presentations plus the same number of teacher presentations;
this is experiment-wide cost, not a single25M-token training trajectory.
Optional --pilot-updates changes the prespecified per-arm budget uniformly;
changing it after starting requires a separate contract/namespace, not an exact resume.

One GPU, sequential x1/x3/x10 child processes; one active optimizer at a time.
Layer-streamed exact end-to-end chain rule and host-staged hidden activations,
not monolithic whole-model backward and not independent layer training.
Mandatory planner measures physical batches1..8 on fixed q/Adam; require batch8
to fit in every arm even if a smaller batch benchmarks faster. No silent batch
change. Actual batch8 gradient audited against mean fixed-q single-example
gradients; probe updates are discarded and probe token presentations separate.

Evaluate A/B/C Wiki, D C4, E Edu at source and registered update points0/128/512/
1024/2048/4096, plus final CURRENT. English stories and Russian Wikipedia16-window
progress,128-window final. The source166128-window fresh panels are recorded in
the immutable descriptor for like-for-like final comparisons. All these panels
have been inspected and are diagnostic validation, not a clean final benchmark.
BEST bookkeeping restarts from CURRENT181145 per arm; original q/Adam untouched.

Primary comparison: final Edu E at equal accepted tokens, with D/A/B/C and
EN/RU transfer shown alongside. Aggregate only ranks E if all three budgets and
final diagnostics complete with no pending state/report sync. E ranking alone
does not establish an overall winner. No statistical superiority or multiple-seed
claim from this single-seed123 screen. No automatic winner continuation yet.

Wall budget is shared across the entire launch; remaining time is divided among
remaining arms. Each arm reserves18min inside its allocation for finalization.
On a deadline or soft STOP, preserve accepted q/Adam/cursor, stop scheduling arms,
and mark comparison partial. A partial x10 cannot be compared to a full x1 tail.
Nonfinite proposals are rejected; unexpected failure aborts the campaign with
saved diagnostics, not hidden by an automatic LR/batch reduction.

Persistence: separate states/runtime-lr-x1,x3,x10, one pruned current generation
each; local every30min and exit. No initial22GB state copy for mere preflight.
Full states HF at each arm exit, small reports during work; about22GB per arm,
96GB declared total upload budget split32GB/arm. Remote namespaces
experiments/exp167-lr-sweep-v1/x1,x3,x10. Source descriptor and parent manifest
stored separately to avoid recursive metadata growth. Never delete parent or
unrelated jobs/checkpoints. Local default cap220GB, minimum free20GB; upfront
check allows up to4x22GB additional storage (three currents plus atomic temporary).
Startup GPU memory budget leaves5GiB plus2GiB allowance for non-JAX CUDA use;
other services can change memory usage later, so it is not a hard GPU reservation.

One existing read-only Telegram listener auto-registers the active arm's report,
then aggregate report on completion. At most start/finish notices per arm;
no per-step Telegram messages. Shared work-dir STOP and SIGTERM support soft
stopping; SIGKILL can lose updates after the last local durable checkpoint.

Engineering validation:21 unique CPU tests across new LR, existing full-stream
and wrapper suites. Includes actual carried-Adam update scaling and two steps,
source chunk restore without bootstrap copy, immutable source descriptors,
all-batch/fixed8 planner, no unequal-tail ranking, serial orchestration and routes.
Physical Blackwell throughput/HBM, full model quality, live Telegram and remote
state transfers remain to be measured on the server.

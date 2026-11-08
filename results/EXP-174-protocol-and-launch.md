# EXP174 · Matched learning-rate selection and continuation

EXP173 used peak LR 1e-4. On four held-out Edu windows, its gqa7 student went from NLL 2.5812 to 7.9543 after 128 updates. This motivates a lower-rate search from the clean EXP171 CURRENT5120 checkpoint. The earlier run alone cannot isolate the effect of LR.

## Design

| Student weights trained | Peak learning rates |
| --- | --- |
| All 21 Mamba mixers | 1e-6, 3e-6, 1e-5 |
| All active student weights, including retained attention, MLPs, norms and the tied embedding/head | 1e-6, 3e-6, 1e-5 |

Both scopes use the 7-attention student and a separate fixed Qwen3-1.7B-Base teacher. The objective is 0.5 next-token CE + 0.5 exact full-vocabulary forward KL at temperature 1. Each pilot starts with fresh Adam, warms up for 32 updates, then holds its assigned LR.

The same 128 updates per pilot present 2,097,152 input tokens at context 1,024 and effective batch 16. Training starts at token 83,886,080 in the verified 600M FineWeb-Edu pack, matching EXP173's training start and replaying no tokens within a branch. Pilot costs are reported separately from the winning trajectory.

Physical batches 8 and 16 are timed separately for each scope in disposable workers. Both process the same 16 sequences per optimizer update. The supervisor uses the faster verified fit for that scope. Failures and timeouts remain in the report and do not prevent trials for the other scope.

The four historical windows track the curve. The minimum endpoint NLL on 16 separate held-out windows, indices 32 through 47 in 1,024-token units, selects the best completed pilot. The report includes the starting NLL on those windows and all completed endpoints. A higher NLL than the starting point does not stop a pilot or continuation.

The winner carries its parameters, Adam moments and data cursor forward at the selected LR. There is no target NLL or continuation step cap. The time limit reserves 18 minutes for a Mamba-only save or 30 minutes for a full-student save. A manual STOP, numerical/runtime failure or exhaustion of fresh corpus can also end the run.

## Kaggle

Use `notebooks/extent-tpu-174.ipynb` with TPU v5e-8, internet access and the existing deploy key, HF and Telegram secrets. The nine-hour budget includes installation, trials and saving. Large artifacts stay in tmpfs during the session. Only the winner is synced to HF under `experiments/exp174-gqa7-lr-v1/<candidate>`.

Send back `extent-results/extent-m3q-tpu174.json` from Kaggle Output. Worker logs are in the same directory. The final report separates probe speeds, accepted updates, validation NLL and HF verification. A new invocation can resume a saved winner with `--resume-candidate <winner> --batch <selected_batch>` using the same code and corpus.

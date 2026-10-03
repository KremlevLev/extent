# EXP-100: CE-first weak anchoring

Prepared2026-10-03, no TPU outcomes observed.

Question: does a small prediction anchor improve stable protected32 correction
learning while keeping true-token CE as the main objective?

All arms protected INOUT rank32, Adam3e-4, stable clip1, unchanged EXP072-v2
seeds123/456. Control `CE`. Primary `QWEN-ANCHOR` = CE+0.1*forward_KL(original
Qwen||student,T=2). Secondary `START-ANCHOR` = CE+0.1*forward_KL(unchanged
ONPOLICY||student,T=2). The existing forward_KL includes multiplication byT²;
targets are stop_gradient and teacher/start parameters never train.
No delta-MSE stage, no scheduled loss switching, no inference-time teacher.

Horizons2,048/4,096/8,192;49,152 total steps,12,582,912 student input tokens.
All six must complete. At both seeds QWEN-ANCHOR must beat CE by at least0.1
final Wiki test NLL, and both primary and CE must improve their own starts.
PG19 confirmation is secondary. START-ANCHOR cannot replace primary post hoc.

EXP095's teacher objective was KL+0.1CE, not this CE-first mixture. Its
self-anchor formula already existed; the new test combines that secondary
objective with protected32/stable clipping and fresh reserved ranges.
Teacher/anchor extra forwards are charged as additional compute, not hidden
behind equal student token counts. Buffers/throughput on real TPU are unmeasured.

Shared source/data/evaluation/resume/budget contract and ready Kaggle cell:
[three-account protocol](EXP-098-100-three-account-protocol.md).
Implementation: `scripts/m3q_weak_anchor_campaign.py`, shared safe engine.
Default8h includes45min reserve; fixed endpoint resumes if the session is partial.

# Local validation, 19 October 2026

Prepared only. No physical TPU/T4 session launched here.

- 33 CPU tests pass in183.77s with eight virtual CPU devices:
  `tests/test_continuation_matrix.py`, `tests/test_hf_download_retry.py`,
  `tests/test_hf_artifact_sync.py`.
- Real small hybrid model forward, fold/new ranks32/64/128/256, learned pregate
  preservation, finite gradient/optimizer steps, six Adam counters and actual
  checkpoint roundtrip. TPU path uses an eight-device mesh; GPU path one device.
- Actual runner startup through first accepted update and deadline save for149
  and151; deliberately mismatched source NLL is rejected before any update.
- Real reference gradient calculations exercise compiled parity/HBM fallback;
  failed compiled candidate is not accepted. CLI dispatches TPU jobs in process,
  GPU jobs to two isolated workers with notifications disabled in children.
- HF tests include repeated429, permanent error rejection, bounded wait, shared
  cooldown and distinct source/save-verification deadlines.
- REAL147current31126 payload downloaded and SHA-checked:487 chunks,
  114,979,740bytes; source q/Adam/best restored. Six counters18540, best31109,
  7,182,336 coordinate elements, empty GQA dictionaries6/13/20/27 retained.
  CPU synthetic-gradient next optimizer update matches original147 bitwise,
  all six counters18541. This is not a full1.7B model-gradient/forward benchmark.
  See EXP-149-154-source-audit.json.
- All original143–148 contracts checked against frozen plans/cloud reports and
  unchanged. Six launch-cell Python bodies parse; hardware budgets and new plans
  validated. Engineering-plan JSON records the exact new code/descriptor hashes.

These checks establish local implementation evidence, not cloud throughput,
full-model parity on TPU, rank superiority, plateau causality or recovery of
teacher quality. First source-window replay and real gradient audit execute on
the actual TPU session before accepting updates. 25–30M tokens is not guaranteed;
check the recorded selected engine and measured throughput.

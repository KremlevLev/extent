# Kaggle Operations Runbook

This file is operational documentation, not part of the scientific report or
paper. Kaggle accelerator sessions are assumed to have a 10-hour ceiling.

## Session timing

- Treat 9 hours as the training deadline for a session.
- Stop launching long evaluations or new compiled shapes after 8 h 30 min.
- Begin the final checkpoint no later than 9 h 10 min.
- Reserve the remaining time for checkpoint validation, upload/export, logs,
  and a Telegram success or failure notification.

## Recoverable checkpoint contract

Every resumable checkpoint must contain:

- model parameters and Lion momentum;
- global step and processed-token count;
- learning-rate schedule state;
- JAX training, dropout, and data-order RNG state;
- dataset revision, shard index, sample/token cursor, and shuffle seed;
- model config, mesh, precision, source revision, and code commit;
- latest training/evaluation metrics and checkpoint checksum.

Write checkpoints atomically to a temporary sibling directory, validate that
all expected leaves exist, then promote the completed directory. A checkpoint
is not durable merely because it exists under `/kaggle/working`; before the
session ends it must be exported through a persisted Kaggle output/dataset or
downloaded to other durable storage.

## Resume gate

Before the first resumed update, print and verify the checkpoint commit, config
hash, global step, processed-token count, data cursor, and optimizer state. Run
one finite-loss step and confirm that the counter advances exactly once. Never
silently restart the scheduler or data stream from zero.

## Planned automation

The full trainer will checkpoint periodically and on a time-budget stop signal,
retain at least the latest two validated checkpoints, update a small manifest,
and send Telegram notifications for start, periodic progress, final save,
failure, and successful resume.

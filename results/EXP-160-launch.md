# EXP160 — CPU corpus preparation, then existing T4 notebook

## Automatic last-cell launch for a fresh12h two-T4 session

No manual dataset upload is required. Existing setup cells provide dependencies,
HF secrets and the repository. Internet must be enabled. The last cell is:

```python
from scripts.m3q_launch_160 import main
result = main()
```

This builds the registered100M C4 on CPU if absent and verifies its exact bytes
against the tracked manifest. It then starts EXP160 with the REMAINING time.
The12h clock starts at the current Jupyter kernel's creation, includes corpus
preparation, and retains5min platform margin plus18min inside-run save reserve.
Existing setup time is subtracted too. An interactive kernel reused for more
than the available session time is rejected; use a fresh session for this cell.
No new notebook. No automatic training on a partial or changed corpus.

The pre-mounted method below avoids preparing C4 inside the allocated GPU session.

Prepare once on a CPU machine with the repository environment (transformers,
huggingface_hub, tokenizers, numpy, filelock). It does not require model weights:

```python
from scripts.prepare_broad_corpus import main
manifest = main([
    "--output", "/path/to/extent-c4-en-100m-v1",
    "--train-tokens", "100000000",
    "--validation-windows", "512",
    "--max-hours", "6",
    "--max-local-gb", "8",
])
```

Restart SAME command after transient interruption. Do not move token files without
documents.sqlite until manifest.json exists. Copy completed directory to the
training host or attach as a Kaggle dataset. Training needs manifest.json,
train.bin and validation.bin only; tokenizer/SQLite are preparation provenance.
For further preparation/resume retain the original full directory on CPU host.
No new notebook is required. Corpus bytes are deliberately not in Git.

Last cell of the existing two-T4 notebook AFTER mounting the prepared corpus:

```python
from scripts.m3q_corpus_recovery_160 import main
result = main([
    "--corpus-dir", "/kaggle/input/extent-c4-en-100m-v1",
    "--max-wall-hours", "11.8",
    "--save-reserve-minutes", "18",
    "--max-steps", "262144",
    "--max-new-hf-gb", "8",
])
```

Change corpus path to the ACTUAL directory containing manifest.json. Plan check
before booking GPU uses the same cell plus "--plan-only" (no model load).
Runner requires existing HF credentials like159. Start/finish Telegram only.
One result to send: extent-m3q-corpus-recovery-160.json.

GPU0 intentionally retains old Wiki as a data control; GPU1 uses new C4. This is
a new experiment from118304, not continuation of159 and not a MOHAWK replication.
No launch is implied by the existence of these cells.

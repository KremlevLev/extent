from __future__ import annotations

from pathlib import Path
import shutil
import time
from typing import Iterable
from urllib.request import urlopen

import numpy as np


WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILES = {
    split: f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    for split in ("train", "validation", "test")
}
WIKITEXT_TRAIN_FILE = WIKITEXT_FILES["train"]
PG19_REPO = "deepmind/pg19"
PG19_REVISION = "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"
PG19_ASSET_ROOT = "https://storage.googleapis.com/deepmind-gutenberg"
WIKITEXT_CONFIG_FILES = {
    "wikitext-2-raw-v1": {
        split: (filename,) for split, filename in WIKITEXT_FILES.items()
    },
    "wikitext-103-raw-v1": {
        "train": (
            "wikitext-103-raw-v1/train-00000-of-00002.parquet",
            "wikitext-103-raw-v1/train-00001-of-00002.parquet",
        ),
        "validation": ("wikitext-103-raw-v1/validation-00000-of-00001.parquet",),
        "test": ("wikitext-103-raw-v1/test-00000-of-00001.parquet",),
    },
}


def pack_tokenized_texts(
    tokenizer,
    texts: Iterable[str],
    total_tokens: int,
    *,
    token_offset: int = 0,
) -> np.ndarray:
    """Pack non-empty documents with EOS separators deterministically."""
    if total_tokens < 1:
        raise ValueError("total_tokens must be positive")
    if token_offset < 0:
        raise ValueError("token_offset must be non-negative")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must define eos_token_id")
    pieces: list[int] = []
    for text in texts:
        if not text or not text.strip():
            continue
        pieces.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        pieces.append(int(eos))
        required = token_offset + total_tokens
        if len(pieces) >= required:
            return np.asarray(
                pieces[token_offset:required], dtype=np.int32
            )
    raise ValueError(
        f"dataset supplied only {len(pieces)} of {token_offset + total_tokens} "
        "required prefix tokens"
    )


def load_wikitext2_tokens(
    total_tokens: int,
    cache_dir: str | Path,
    *,
    tokenizer_repo: str,
    tokenizer_revision: str,
    token_offset: int = 0,
    dataset_split: str = "train",
    dataset_config: str = "wikitext-2-raw-v1",
) -> np.ndarray:
    """Load a pinned raw WikiText configuration with a pinned tokenizer."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dataset_config not in WIKITEXT_CONFIG_FILES:
        raise ValueError(f"unsupported WikiText configuration: {dataset_config}")
    config_files = WIKITEXT_CONFIG_FILES[dataset_config]
    if dataset_split not in config_files:
        raise ValueError(
            f"unsupported {dataset_config} split: {dataset_split}"
        )
    dataset_files = [
        hf_hub_download(
            repo_id=WIKITEXT_REPO,
            repo_type="dataset",
            revision=WIKITEXT_REVISION,
            filename=filename,
            cache_dir=cache_dir,
        )
        for filename in config_files[dataset_split]
    ]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        revision=tokenizer_revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    texts = []
    for dataset_file in dataset_files:
        texts.extend(
            parquet.read_table(dataset_file, columns=["text"])["text"].to_pylist()
        )
    return pack_tokenized_texts(
        tokenizer, texts, total_tokens, token_offset=token_offset
    )


def load_pg19_tokens(
    total_tokens: int,
    cache_dir: str | Path,
    *,
    tokenizer_repo: str,
    tokenizer_revision: str,
    token_offset: int = 0,
    dataset_split: str = "validation",
) -> np.ndarray:
    """Load a deterministic PG-19 slice from its pinned split manifest."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    if dataset_split not in ("train", "validation", "test"):
        raise ValueError(f"unsupported PG-19 split: {dataset_split}")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = hf_hub_download(
        repo_id=PG19_REPO,
        repo_type="dataset",
        revision=PG19_REVISION,
        filename=f"data/{dataset_split}_files.txt",
        cache_dir=cache_dir,
    )
    names = sorted(
        line.strip() for line in Path(manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        revision=tokenizer_revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )

    def texts():
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe PG-19 manifest path: {name}")
            local = cache_dir / "pg19-assets" / relative
            if not local.exists():
                local.parent.mkdir(parents=True, exist_ok=True)
                staging = local.with_suffix(local.suffix + ".tmp")
                retry_delays = (2, 5, 15, 30, 60)
                for attempt in range(len(retry_delays) + 1):
                    try:
                        with urlopen(f"{PG19_ASSET_ROOT}/{name}", timeout=120) as source:
                            with staging.open("wb") as target:
                                shutil.copyfileobj(source, target)
                        if staging.stat().st_size == 0:
                            raise IOError(f"empty PG-19 asset: {name}")
                        staging.replace(local)
                        break
                    except Exception:
                        staging.unlink(missing_ok=True)
                        if attempt == len(retry_delays):
                            raise
                        delay = retry_delays[attempt]
                        print(
                            f"pg19_download=RETRY asset={name} wait_seconds={delay}",
                            flush=True,
                        )
                        time.sleep(delay)
            yield local.read_text(encoding="utf-8")

    return pack_tokenized_texts(
        tokenizer, texts(), total_tokens, token_offset=token_offset
    )

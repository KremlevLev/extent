from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILES = {
    split: f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    for split in ("train", "validation", "test")
}
WIKITEXT_TRAIN_FILE = WIKITEXT_FILES["train"]


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
) -> np.ndarray:
    """Load a pinned WikiText-2 train slice with a pinned tokenizer."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dataset_split not in WIKITEXT_FILES:
        raise ValueError(
            f"unsupported WikiText-2 split: {dataset_split}"
        )
    dataset_file = hf_hub_download(
        repo_id=WIKITEXT_REPO,
        repo_type="dataset",
        revision=WIKITEXT_REVISION,
        filename=WIKITEXT_FILES[dataset_split],
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        revision=tokenizer_revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    texts = parquet.read_table(dataset_file, columns=["text"])["text"].to_pylist()
    return pack_tokenized_texts(
        tokenizer, texts, total_tokens, token_offset=token_offset
    )

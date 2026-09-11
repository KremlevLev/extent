import numpy as np

from extent.calibration_data import (
    PG19_REVISION,
    WIKITEXT_CONFIG_FILES,
    WIKITEXT_FILES,
    pack_tokenized_texts,
)


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def test_pack_tokenized_texts_skips_blanks_adds_eos_and_truncates():
    tokens = pack_tokenized_texts(_Tokenizer(), ["", "ab", " ", "cd"], 5)
    np.testing.assert_array_equal(tokens, [97, 98, 99, 99, 100])


def test_pack_tokenized_texts_supports_locked_token_offset():
    tokens = pack_tokenized_texts(
        _Tokenizer(), ["ab", "cd", "ef"], 4, token_offset=3
    )
    np.testing.assert_array_equal(tokens, [99, 100, 99, 101])


def test_wikitext_split_files_are_pinned_explicitly():
    assert WIKITEXT_FILES["validation"] == (
        "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
    )
    assert WIKITEXT_CONFIG_FILES["wikitext-103-raw-v1"]["validation"] == (
        "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
    )
    assert PG19_REVISION == "4d28bd77e66947ad3835cf78ed7aaeb4dd87ad8b"

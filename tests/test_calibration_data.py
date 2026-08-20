import numpy as np

from extent.calibration_data import pack_tokenized_texts


class _Tokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def test_pack_tokenized_texts_skips_blanks_adds_eos_and_truncates():
    tokens = pack_tokenized_texts(_Tokenizer(), ["", "ab", " ", "cd"], 5)
    np.testing.assert_array_equal(tokens, [97, 98, 99, 99, 100])

from typing import List, Dict, Optional
from transformers import AutoTokenizer


class QwenTokenizer:
    """
    Обёртка над токенизатором Qwen3.
    Загружает токенизатор, добавляет pad_token, предоставляет encode/decode.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        """
        Токенизирует текст.
        Возвращает словарь с input_ids и attention_mask.
        """
        return self.tokenizer(
            text,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Декодирует токены обратно в текст."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

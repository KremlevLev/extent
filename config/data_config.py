from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DataConfig:
    """
    Параметры данных для continue pre-training.
    """
    # --- Пути к датасетам ---
    train_data_paths: List[str] = field(default_factory=lambda: [
        "gs://my-bucket/data/web_text.jsonl",
        "gs://my-bucket/data/code.jsonl",
        "gs://my-bucket/data/math.jsonl",
        "gs://my-bucket/data/science.jsonl",
        "gs://my-bucket/data/books.jsonl",
        "gs://my-bucket/data/multilingual.jsonl",
        "gs://my-bucket/data/synthetic.jsonl",
    ])
    eval_data_paths: Optional[List[str]] = None

    # --- Пропорции смешивания датасетов ---
    # web(35%), code(15%), math(10%), science(10%), books(10%), multilingual(15%), synthetic(5%)
    dataset_mixing_ratios: List[float] = field(default_factory=lambda: [0.35, 0.15, 0.10, 0.10, 0.10, 0.15, 0.05])

    # --- Токенизация ---
    tokenizer_path: str = "Qwen/Qwen3-14B"
    max_seq_length: int = 8192
    padding: str = "max_length"  # "max_length", "longest", False
    truncation: bool = True

    # --- Загрузка ---
    num_workers: int = 8
    prefetch_factor: int = 2
    shuffle_buffer_size: int = 10000
    seed: int = 42

    # --- Кэширование ---
    cache_dir: Optional[str] = None
    use_cache: bool = True

    # --- Фильтрация ---
    min_length: int = 50
    max_length: int = 8192
    remove_duplicates: bool = True

    # --- Формат данных ---
    data_format: str = "jsonl"  # "jsonl", "parquet", "text"
    text_key: str = "text"      # Ключ в JSON, содержащий текст

    def __post_init__(self):
        assert len(self.train_data_paths) == len(self.dataset_mixing_ratios), \
            f"Количество путей ({len(self.train_data_paths)}) должно совпадать с количеством пропорций ({len(self.dataset_mixing_ratios)})"
        assert abs(sum(self.dataset_mixing_ratios) - 1.0) < 1e-6, \
            f"Сумма пропорций должна быть 1.0, а не {sum(self.dataset_mixing_ratios)}"

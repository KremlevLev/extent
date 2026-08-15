from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """
    Гиперпараметры обучения гибридной модели.
    """
    # --- Основные параметры ---
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = field(init=False)  # batch_size * gradient_accumulation_steps * num_tpu_cores
    max_steps: int = 100_000
    max_tokens: Optional[int] = 40_000_000_000  # 40B токенов

    # --- Learning rate ---
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant", "warmup_stable_decay"
    warmup_type: str = "linear"

    # --- Оптимизатор ---
    optimizer: str = "lion"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # --- Точность ---
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True

    # --- Чекпоинты и логирование ---
    save_steps: int = 1000
    save_total_limit: int = 5
    eval_steps: int = 500
    logging_steps: int = 10
    output_dir: str = "./checkpoints"

    # --- TPU ---
    num_tpu_cores: int = 8
    data_parallel_shards: int = 8
    model_parallel_shards: int = 1
    sequence_parallel: bool = False

    # --- Прочее ---
    seed: int = 42
    use_ema: bool = False
    ema_decay: float = 0.9999
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps * self.num_tpu_cores

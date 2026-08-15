"""JAX research prototype for a Qwen -> Mamba-3/MLA hybrid."""

from .config import HybridConfig, tiny_config
from .model import HybridForCausalLM

__all__ = ["HybridConfig", "HybridForCausalLM", "tiny_config"]

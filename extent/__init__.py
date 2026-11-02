"""JAX research prototype for a Qwen -> Mamba-3/MLA hybrid."""

from .config import HybridConfig, tiny_config

def __getattr__(name):
    # PyTorch-only tools must not initialize/import the JAX stack.
    if name == "HybridForCausalLM":
        from .model import HybridForCausalLM
        return HybridForCausalLM
    raise AttributeError(name)

__all__ = ["HybridConfig", "HybridForCausalLM", "tiny_config"]

from .common import RMSNorm, SwiGLU
from .mamba3 import Mamba3MIMO
from .mla import MultiHeadLatentAttention

def __getattr__(name):
    # qwen3_teacher imports common while rorope_bkv imports qwen3_teacher.
    if name == "Qwen3RoRoPEBKVAttention":
        from .rorope_bkv import Qwen3RoRoPEBKVAttention
        return Qwen3RoRoPEBKVAttention
    raise AttributeError(name)

__all__ = [
    "Mamba3MIMO",
    "MultiHeadLatentAttention",
    "Qwen3RoRoPEBKVAttention",
    "RMSNorm",
    "SwiGLU",
]

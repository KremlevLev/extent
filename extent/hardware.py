from __future__ import annotations

from dataclasses import dataclass

import jax


@dataclass(frozen=True)
class ComputeDTypeDecision:
    dtype: str
    reason: str


_PRE_AMPERE_GPU_MARKERS = (
    "T4",
    "V100",
    "P100",
    "P40",
    "P4",
    "K80",
)


def recommended_compute_dtype(
    devices: list[jax.Device] | None = None,
    requested: str = "auto",
) -> ComputeDTypeDecision:
    """Choose a safe activation dtype without changing parameter storage.

    Pre-Ampere NVIDIA GPUs do not have native BF16 Tensor Cores. In particular,
    the sharded BF16 GEMM backward on Kaggle T4 can emit values near BF16 max.
    TPU v5e and modern BF16-capable accelerators retain BF16 compute.
    """
    if requested != "auto":
        return ComputeDTypeDecision(requested, "explicit command-line override")
    devices = list(jax.devices() if devices is None else devices)
    kinds = tuple(str(getattr(device, "device_kind", device)) for device in devices)
    if any(device.platform == "gpu" for device in devices) and any(
        marker.lower() in kind.lower() for marker in _PRE_AMPERE_GPU_MARKERS for kind in kinds
    ):
        return ComputeDTypeDecision(
            "float32",
            f"pre-Ampere GPU detected ({', '.join(sorted(set(kinds)))})",
        )
    return ComputeDTypeDecision("bfloat16", "BF16-capable or non-GPU accelerator")

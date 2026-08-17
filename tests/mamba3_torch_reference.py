"""Independent eager PyTorch oracle for the official Mamba-3 MIMO step."""

from __future__ import annotations

import math

import torch


MAMBA3_REFERENCE_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def _rotate_mimo_halves(
    value: torch.Tensor, angle: torch.Tensor, rotary_pairs: int
) -> torch.Tensor:
    half = value.shape[-1] // 2
    real, imaginary = value[..., :half], value[..., half:]
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    while cosine.ndim < real.ndim:
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
    if rotary_pairs < half:
        cosine = torch.nn.functional.pad(cosine, (0, half - rotary_pairs), value=1.0)
        sine = torch.nn.functional.pad(sine, (0, half - rotary_pairs), value=0.0)
    return torch.cat(
        (real * cosine - imaginary * sine, real * sine + imaginary * cosine),
        dim=-1,
    )


def torch_mamba3_mimo_scan(
    x: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    decay: torch.Tensor,
    trap: torch.Tensor,
    angle_step: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_z: torch.Tensor,
    mimo_o: torch.Tensor,
    skip: torch.Tensor,
    rotary_pairs: int,
) -> torch.Tensor:
    """Direct recurrence matching the official one-token MIMO kernel semantics."""
    batch, length, heads, head_dim = x.shape
    rank, state_dim = b.shape[2], b.shape[-1]
    state = torch.zeros(batch, heads, head_dim, state_dim, dtype=torch.float32)
    previous_k = torch.zeros(batch, rank, heads, state_dim, dtype=torch.float32)
    previous_v = torch.zeros(batch, heads, head_dim, dtype=torch.float32)
    angle = torch.zeros(batch, heads, rotary_pairs, dtype=torch.float32)
    rank_first_x = mimo_x.transpose(0, 1).float()
    rank_first_z = mimo_z.transpose(0, 1).float()
    rank_first_o = mimo_o.transpose(0, 1).float()
    outputs = []

    for index in range(length):
        dt_t = dt[:, index].float()
        alpha = torch.exp(decay[:, index].float() * dt_t)
        trap_t = torch.sigmoid(trap[:, index].float())
        gamma = trap_t * dt_t
        beta = (1.0 - trap_t) * dt_t * alpha
        angle = angle + math.pi * angle_step[:, index, None].float() * dt_t[..., None]
        b_rot = _rotate_mimo_halves(b[:, index].float(), angle, rotary_pairs)
        c_rot = _rotate_mimo_halves(c[:, index].float(), angle, rotary_pairs)
        value_now = x[:, index, None].float() * rank_first_x[None]
        value_previous = previous_v[:, None] * rank_first_x[None]
        injection = torch.einsum(
            "brhp,brhn->bhpn", gamma[:, None, :, None] * value_now, b_rot
        )
        previous = torch.einsum(
            "brhp,brhn->bhpn",
            beta[:, None, :, None] * value_previous,
            previous_k,
        )
        state = alpha[:, :, None, None] * state + injection + previous
        y_rank = torch.einsum("bhpn,brhn->brhp", state, c_rot)
        y_rank = y_rank + skip[None, None, :, None].float() * value_now
        gate = torch.nn.functional.silu(
            z[:, index, None].float() * rank_first_z[None]
        )
        outputs.append(torch.sum(y_rank * gate * rank_first_o[None], dim=1))
        previous_k = b_rot
        previous_v = x[:, index].float()

    return torch.stack(outputs, dim=1).to(x.dtype)

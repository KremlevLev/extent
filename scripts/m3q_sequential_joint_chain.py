"""Finish EXP-072, then use the same TPU session for EXP-073."""
from __future__ import annotations

import gc
import jax

from scripts.m3q_full_depth_sequential_confirmation import main as run_exp072
from scripts.m3q_sequential_joint_recovery_campaign import main as run_exp073


def main(argv=None):
    exp072 = run_exp072([])
    if exp072.get("status") != "completed":
        return {"status": "exp072_incomplete", "exp072": exp072, "exp073": None}
    jax.clear_caches()
    gc.collect()
    exp073 = run_exp073(list(argv or ()))
    return {"status": exp073.get("status"), "exp072": exp072, "exp073": exp073}


if __name__ == "__main__":
    main()

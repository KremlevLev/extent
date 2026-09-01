"""Short Kaggle entry point for EXP-068."""

from __future__ import annotations

from scripts.m3q_compatibility_atlas_campaign import main as run_atlas


def main(argv: list[str] | None = None) -> dict:
    return run_atlas(["--long-horizon-confirmation", *(argv or [])])


if __name__ == "__main__":
    main()

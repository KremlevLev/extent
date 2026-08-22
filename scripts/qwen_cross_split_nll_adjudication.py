from __future__ import annotations

import sys

from scripts.qwen_streamed_multiseed_end_to_end import main as run_protocol


def main(argv: list[str] | None = None) -> dict:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--protocol" in arguments:
        raise ValueError("EXP-043 wrapper fixes the protocol internally")
    return run_protocol(["--protocol", "exp043-validation", *arguments])


if __name__ == "__main__":
    main()

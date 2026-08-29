from __future__ import annotations

import argparse
import json
from pathlib import Path

from extent.config import load_config
from extent.hybrid_transplant import build_hybrid_transplant_plan
from extent.qwen_source import QWEN3_14B, validate_source_marker, validate_source_metadata


def _validate_local_metadata(model_dir: Path, expected_mixer_tensors: set[str]) -> None:
    validate_source_marker(model_dir, QWEN3_14B)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    validate_source_metadata(config, index, QWEN3_14B)
    missing = expected_mixer_tensors - set(index["weight_map"])
    if missing:
        raise KeyError(f"checkpoint index is missing mixer tensors: {sorted(missing)[:3]}")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(
        description="Shape-only full Qwen3 -> Extent-14B transplant readiness audit."
    )
    parser.add_argument("--config", default="config/hybrid_14b_v5e8.yaml")
    parser.add_argument("--qwen-model-dir")
    parser.add_argument(
        "--result-json",
        default="output/full-hybrid-transplant-preflight.json",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return an error while the production architecture has blockers.",
    )
    args = parser.parse_args(argv)

    config, _ = load_config(args.config)
    plan = build_hybrid_transplant_plan(config)
    metadata_validated = False
    if args.qwen_model_dir:
        expected_mixer_tensors = {
            name for action in plan.actions for name in action.source_tensors
        }
        _validate_local_metadata(Path(args.qwen_model_dir), expected_mixer_tensors)
        metadata_validated = True

    result = plan.to_dict()
    result["checkpoint_metadata_validated"] = metadata_validated
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"source={plan.source}")
    print(
        f"layer_plan=PASS attention={len(plan.attention_layer_indices)}/40 "
        f"({plan.attention_fraction:.0%}) mamba={len(plan.mamba_layer_indices)}/40"
    )
    print(
        f"source_ownership=PASS direct={plan.direct_tensor_count} "
        f"mixer={plan.mixer_tensor_count} total={plan.source_tensor_count}/443"
    )
    print(f"mamba_method={plan.mamba_method} layers={len(plan.mamba_layer_indices)}")
    print(
        f"mla_method={plan.mla_method} evidence={plan.mla_evidence_experiment} "
        f"layers={len(plan.attention_layer_indices)}"
    )
    print(
        "retained_attention_cache="
        f"{plan.source_retained_attention_cache_elements_per_token}->"
        f"{plan.target_retained_attention_cache_elements_per_token} elements/token "
        f"reduction={plan.retained_attention_cache_reduction_fraction:.3%}"
    )
    if plan.architecture_ready:
        print("verdict=GO: production modules match every selected transplant")
    else:
        print("verdict=NO-GO: mixer methods are selected, but production MLA is incompatible")
        for blocker in plan.blockers:
            print(f"  blocker={blocker}")
    print(f"result_json={result_path.resolve()}")

    if args.require_ready and not plan.architecture_ready:
        raise RuntimeError("full hybrid transplant is not ready; see blockers above")
    return result


if __name__ == "__main__":
    main()

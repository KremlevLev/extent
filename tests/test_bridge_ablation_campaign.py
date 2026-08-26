from __future__ import annotations

from pathlib import Path

from scripts.qwen_bridge_ablation import (
    ARM_ORDER,
    _checkpoint_steps,
    aggregate_bridge_screen,
)
from scripts.qwen_bridge_ablation_campaign import (
    RECOVERY_STEPS,
    TARGET_LAYERS,
    _cache_arguments,
    render_summary,
)
from scripts.qwen_stabilized_bridge_campaign import (
    CAMPAIGN_PROTOCOL as STABILIZED_CAMPAIGN_PROTOCOL,
    DEFAULT_ARGUMENTS as STABILIZED_DEFAULT_ARGUMENTS,
    LAYER_PROTOCOL as STABILIZED_LAYER_PROTOCOL,
)


def _recovery(value: float) -> dict:
    return {
        "complete": True,
        "completed_steps": RECOVERY_STEPS,
        "evaluations": {
            str(RECOVERY_STEPS): {
                "decoder_output": {"relative_l2": value}
            }
        },
    }


def _layer_result(offset: float) -> dict:
    seeds = {}
    for index, seed in enumerate((123, 456, 789)):
        random = offset + 0.50 + index * 0.01
        seeds[str(seed)] = {
            "arms": {
                "CONTROL-RANDOM": {"recovery": _recovery(random)},
                "APPLE-BRIDGE": {"recovery": _recovery(random - 0.01)},
                "BRIDGE-PLUS-ORIENTATION": {
                    "recovery": _recovery(random - 0.03)
                },
                "MOHAWK-ORIENTATION": {
                    "recovery": _recovery(random - 0.02)
                },
                "CONTROL-QKVO": {"recovery": _recovery(random + 0.04)},
            }
        }
    return {"seeds": seeds, "complete": True, "passed": True}


def test_exp054_protocol_prioritizes_representative_layer_and_has_five_arms():
    assert TARGET_LAYERS == (18, 0)
    assert ARM_ORDER[:3] == (
        "CONTROL-RANDOM",
        "APPLE-BRIDGE",
        "BRIDGE-PLUS-ORIENTATION",
    )
    assert len(ARM_ORDER) == 5
    assert _checkpoint_steps("0,256,512,1024", 1024) == (0, 256, 512, 1024)


def test_exp054_aggregate_uses_paired_complete_seeds_and_renders_summary():
    layers = {"18": _layer_result(0.0), "0": _layer_result(0.1)}
    aggregate = aggregate_bridge_screen(layers)
    assert aggregate["complete_layers"] == 2
    assert aggregate["screening_gate_passed"]
    assert aggregate["primary_arm"] == "BRIDGE-PLUS-ORIENTATION"
    assert aggregate["layers"]["18"]["BRIDGE-PLUS-ORIENTATION"][
        "wins_over_random"
    ] == 3
    result = {
        "status": "completed",
        "passed": True,
        "complete": True,
        "duration_hours": 3.0,
        "aggregate": aggregate,
    }
    summary = render_summary(result)
    assert "BRIDGE-PLUS-ORIENTATION" in summary
    assert "Negative `Arm - random` is better" in summary


def test_exp054_ram_cache_keeps_qwen_shards_and_disk_mode_prunes_them(tmp_path: Path):
    common = dict(
        layer=18,
        evaluation_only=False,
        qwen_cache_dir="/dev/shm/qwen",
        dataset_cache_dir="/tmp/data",
        output_dir=tmp_path,
        compute_dtype="bfloat16",
        storage_dtype="float16",
        per_device_windows=4,
    )
    ram_args, ram_manifest, _ = _cache_arguments(qwen_storage="ram", **common)
    disk_args, disk_manifest, _ = _cache_arguments(qwen_storage="disk", **common)
    assert "--prune-consumed-shards" not in ram_args
    assert "--prune-consumed-shards" in disk_args
    assert str(RECOVERY_STEPS) in ram_args
    assert ram_manifest == disk_manifest


def test_exp055_locked_defaults_use_partial_rope_cosine_and_longer_recovery(tmp_path: Path):
    defaults = list(STABILIZED_DEFAULT_ARGUMENTS)
    assert defaults[defaults.index("--campaign-protocol") + 1] == STABILIZED_CAMPAIGN_PROTOCOL
    assert defaults[defaults.index("--layer-protocol") + 1] == STABILIZED_LAYER_PROTOCOL
    assert defaults[defaults.index("--bridge-rope-fraction") + 1] == "0.5"
    assert defaults[defaults.index("--bridge-matrix-loss-weight") + 1] == "0.0"
    assert defaults[defaults.index("--recovery-steps") + 1] == "4096"
    assert defaults[defaults.index("--primary-arm") + 1] == "APPLE-BRIDGE"
    args, manifest, directory = _cache_arguments(
        layer=18,
        evaluation_only=False,
        qwen_cache_dir="/dev/shm/qwen",
        qwen_storage="ram",
        dataset_cache_dir="/tmp/data",
        output_dir=tmp_path,
        compute_dtype="bfloat16",
        storage_dtype="float16",
        per_device_windows=4,
        artifact_prefix="exp055",
        recovery_steps=4096,
    )
    assert "4096" in args
    assert "exp055-layer18-train-cache" in str(directory)
    assert manifest.parent == directory

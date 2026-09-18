"""EXP-088: one session from layout preflight into downstream-sensitive recovery."""
import argparse
import gc
from pathlib import Path
import time
import traceback
import jax

from extent.campaign_checkpoint import write_json_atomic
from extent.hf_artifact_sync import artifact_config_from_env, upload_artifact
from scripts import m3q_pair_threshold_campaign as wrapper
from scripts import m3q_trust_region_sweep_campaign as sweep
from scripts.m3q_joint_pair_campaign import OVERRIDES as PREVIOUS
from scripts.m3q_downstream_preflight import main as run_preflight
from scripts.qwen_extended_horizon_campaign import _safe_notify


OVERRIDES = dict(PREVIOUS, **{
    "PROTOCOL": "exp088-downstream-recovery-v1",
    "HF_PREFIX": "experiments/exp088-downstream",
    "RESULT_STEM": "extent-m3q-downstream",
    "SUMMARY_TITLE": "EXP-088 downstream-sensitive pair recovery",
    "OUTPUT_SUBDIR": "exp088",
    "STATE_PREFIX": "extent-exp088",
    "QWEN_CACHE_DIR": "/dev/shm/qwen3-1.7b-exp088-weights",
    "DERIVED_FROM": "EXP-087 local segment objective comparator failure",
    "PRIMARY_ARM": "DOWNSTREAM-KL", "CONTROL_ARM": "LOCAL-PAIR-MSE",
    "ARM_MIN_RELATIVE_GAIN": {"DOWNSTREAM-KL": 0.001, "LOCAL-PAIR-MSE": 0.001},
    "PRIMARY_PROPOSAL_MODE": "downstream_kl",
    "CONTROL_PROPOSAL_MODE": "joint_segment",
    "INNER_PROTOCOL_PREFIX": "exp088-downstream-replication",
    "INNER_RESULT_PREFIX": "exp088-downstream-replication",
    "INNER_OUTPUT_PREFIX": "exp088/replication",
    "BASE_DATA_SEED": 88_000,
    "BASE_TRAIN_OFFSET": 1_949_696,
    "BASE_SECONDARY_OFFSET": 147_456,
    "BASE_EVAL_OFFSET": 229_376,
})
SWEEP_OVERRIDES = {"STEPS_PER_LAYER": 512, "TRAIN_WINDOWS": 64, "TRAIN_LENGTH": 64}


def configured_contract():
    original = wrapper.OVERRIDES
    old_sweep = {key: getattr(sweep, key) for key in SWEEP_OVERRIDES}
    try:
        wrapper.OVERRIDES = OVERRIDES
        for key, value in SWEEP_OVERRIDES.items():
            setattr(sweep, key, value)
        return wrapper.configured_contract()
    finally:
        wrapper.OVERRIDES = original
        for key, value in old_sweep.items():
            setattr(sweep, key, value)


def run_science(argv):
    original = wrapper.OVERRIDES
    old_sweep = {key: getattr(sweep, key) for key in SWEEP_OVERRIDES}
    try:
        wrapper.OVERRIDES = OVERRIDES
        for key, value in SWEEP_OVERRIDES.items():
            setattr(sweep, key, value)
        return wrapper.main(argv)
    finally:
        wrapper.OVERRIDES = original
        for key, value in old_sweep.items():
            setattr(sweep, key, value)


def validate_runtime_configs():
    base = wrapper.campaign
    original = {key: getattr(base, key) for key in OVERRIDES}
    old_sweep = {key: getattr(sweep, key) for key in SWEEP_OVERRIDES}
    try:
        for key, value in OVERRIDES.items():
            setattr(base, key, value)
        for key, value in SWEEP_OVERRIDES.items():
            setattr(sweep, key, value)
        configs = [base.replication_overrides(rep) for rep in base.REPLICATIONS]
        for config in configs:
            for key in config:
                getattr(sweep, key)
        return configs
    finally:
        for key, value in original.items():
            setattr(base, key, value)
        for key, value in old_sweep.items():
            setattr(sweep, key, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/kaggle/working/output")
    parser.add_argument("--max-wall-hours", type=float, default=8.1)
    parser.add_argument("--telegram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not 3 <= args.max_wall_hours <= 8.25:
        raise ValueError("session budget must be 3-8.25 hours")
    hub = artifact_config_from_env()
    if hub is None:
        raise ValueError("HF configuration required before starting TPU work")
    # Build every replication config before consuming accelerator time.
    configured_contract()
    validate_runtime_configs()
    output = Path(args.output_dir)
    path = output / "extent-m3q-downstream-campaign.json"
    result = {"protocol": "exp088-session-v1", "status": "running", "stage": "preflight"}
    started = time.monotonic()

    def persist():
        result["duration_hours"] = (time.monotonic() - started) / 3600
        write_json_atomic(path, result)
        try:
            upload_artifact(path, "experiments/exp088-downstream/session-latest.json", hub,
                            commit_message="EXP-088 session stage")
        except Exception as exc:
            print(f"session_summary_upload=FAILED type={type(exc).__name__}; local retained", flush=True)

    _safe_notify(args.telegram, "Extent EXP-088 preflight + downstream campaign started")
    try:
        persist()
        preflight = run_preflight(["--output-dir", str(output), "--no-telegram"])
        result["preflight"] = preflight
        if preflight["status"] != "completed":
            raise RuntimeError("preflight not completed; scientific stage must not start")
        jax.clear_caches()
        gc.collect()
        remaining = args.max_wall_hours - (time.monotonic() - started) / 3600
        if remaining < 2:
            result.update(status="deadline_partial", stage="post_preflight")
            return result
        result["stage"] = "scientific_recovery"
        persist()
        scientific = run_science([
            "--output-dir", str(output), "--max-wall-hours", str(remaining), "--no-telegram",
        ])
        result.update(status=scientific["status"], aggregate=scientific["aggregate"],
                      scientific_result=str(output / "exp088" / "extent-m3q-downstream.json"))
        return result
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        persist()
        _safe_notify(args.telegram,
                     f"Extent EXP-088 {result['status']} stage={result['stage']}")


if __name__ == "__main__":
    main()

import jax
from scripts.m3q_downstream_preflight import main


def test_preflight_reuses_aot_executable_for_three_steps(tmp_path):
    result = main(["--tiny", "--no-telegram", "--output-dir", str(tmp_path)])
    assert result["status"] == "completed"
    assert len(result["steps"]) == 3
    assert all(row["grads_finite"] for row in result["steps"])
    assert len(result["devices"]) == jax.device_count()

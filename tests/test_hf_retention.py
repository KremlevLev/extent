from types import SimpleNamespace

from extent.hf_retention import classify_repo_files, is_checkpoint_payload


def file(path, size):
    return SimpleNamespace(path=path, size=size)


def test_retention_keeps_sources_and_scientific_artifacts():
    audit = classify_repo_files([
        file("experiments/exp069-allocation/a/state.msgpack", 100),
        file("experiments/exp069-allocation/a/checkpoint.json", 2),
        file("experiments/exp072-full-depth-sequential-confirmation-v2/a/state.msgpack", 200),
        file("experiments/exp084-robust-consensus/a/state.msgpack", 300),
        file("experiments/exp084-robust-consensus/a/checkpoint.json", 3),
        file("experiments/exp084-robust-consensus/latest.json", 4),
        file("experiments/exp084-robust-consensus/latest-summary.md", 1),
    ])
    assert audit["reviewable_payload_files"] == 1
    assert audit["reviewable_payload_bytes"] == 300
    assert audit["protected_payload_files"] == 2
    assert audit["protected_payload_bytes"] == 300
    assert audit["review_paths"] == [
        "experiments/exp084-robust-consensus/a/checkpoint.json",
        "experiments/exp084-robust-consensus/a/state.msgpack",
    ]


def test_only_stable_checkpoint_payload_name_is_selected():
    assert is_checkpoint_payload("experiments/x/state.msgpack")
    assert not is_checkpoint_payload("experiments/x/training_state.msgpack")
    assert not is_checkpoint_payload("experiments/x/model.safetensors")

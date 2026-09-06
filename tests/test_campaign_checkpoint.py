import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from extent.campaign_checkpoint import CampaignCheckpointStore
from extent.hf_artifact_sync import HubArtifactConfig
from extent.optimizer import create_lion


def test_checkpoint_restores_bf16_lion_and_rejects_mismatched_contract(tmp_path):
    params = {"w": jnp.ones((2, 3), jnp.bfloat16)}
    tx = create_lion(total_steps=4, warmup_steps=1)
    opt = tx.init(params)
    _, opt = tx.update(jax.tree.map(jnp.ones_like, params), opt, params)
    state = {"params": params, "opt_state": opt}
    store = CampaignCheckpointStore(tmp_path, "exp")
    meta = store.save("full", state, contract={"seed": 1}, step=1, metrics={"step": 1, "finite": np.bool_(True)})
    restored, current = store.restore("full", {"seed": 1}, state)
    assert current["step"] == 1
    assert jax.tree.structure(restored) == jax.tree.structure(state)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(restored)):
        np.testing.assert_array_equal(before, after)
        assert before.dtype == after.dtype
    with pytest.raises(ValueError, match="contract mismatch"):
        store.restore("full", {"seed": 2}, state)
    store.save("full", state, contract={"seed": 1}, step=2, metrics={"step": 2})
    assert not (tmp_path / "full" / meta["checkpoint_file"]).exists()
    # Corruption must fail, not trigger silent recomputation.
    latest = store.metadata("full", {"seed": 1})
    (tmp_path / "full" / latest["checkpoint_file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="byte-size"):
        store.restore("full", {"seed": 1}, state)


def test_hub_checkpoint_commits_matching_metadata_payload_with_stable_names(tmp_path, monkeypatch):
    commits = []
    class Api:
        def __init__(self, **kwargs):
            pass
        def create_commit(self, **kwargs):
            commits.append(kwargs)
    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    store = CampaignCheckpointStore(tmp_path, "exp", HubArtifactConfig("user/private", "dataset", "fake"))
    store.save("full", {"x": np.ones(2)}, contract={}, step=3, metrics={})
    assert len(commits) == 1
    ops = commits[0]["operations"]
    assert [op.path_in_repo for op in ops] == ["exp/full/state.msgpack", "exp/full/checkpoint.json"]
    assert json.loads(ops[1].path_or_fileobj)["checkpoint_file"] == "state.msgpack"
    assert commits[0]["repo_type"] == "dataset"
    store.metadata("full", {})
    assert len(commits) == 1

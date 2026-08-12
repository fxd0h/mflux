import json

import pytest

from mflux.models.common.resolution.path_resolution import PathResolution


def _snapshot(tmp_path, shards_present: list[str], referenced: list[str]):
    """A cache snapshot with a sharded transformer whose index names `referenced`."""
    transformer = tmp_path / "transformer"
    transformer.mkdir(parents=True)
    for name in shards_present:
        (transformer / name).write_bytes(b"")
    index = {"weight_map": {f"layer.{i}.weight": name for i, name in enumerate(referenced)}}
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    return tmp_path


@pytest.mark.fast
def test_a_complete_shard_set_is_accepted(tmp_path):
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    snapshot = _snapshot(tmp_path, shards_present=shards, referenced=shards)

    assert PathResolution._is_snapshot_complete(snapshot, {"transformer"}) is True


@pytest.mark.fast
def test_an_interrupted_download_is_not_mistaken_for_a_complete_one(tmp_path):
    # The failure this guards: one shard on disk satisfies "has a safetensors", the loader
    # then updates with strict=False, and the transformer comes up missing weights and
    # generating noise instead of failing.
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    snapshot = _snapshot(tmp_path, shards_present=shards[:1], referenced=shards)

    assert PathResolution._is_snapshot_complete(snapshot, {"transformer"}) is False


@pytest.mark.fast
def test_an_unreadable_index_is_not_treated_as_complete(tmp_path):
    transformer = tmp_path / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "model-00001-of-00002.safetensors").write_bytes(b"")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text("{not json")

    assert PathResolution._is_snapshot_complete(tmp_path, {"transformer"}) is False

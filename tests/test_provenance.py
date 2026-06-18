"""Provenance sidecar manifests (spec Appendix C)."""

import json


def test_manifest_has_required_fields(tmp_path):
    from qsal.provenance import build_manifest, save_manifest

    m = build_manifest(stage="test", damping_lambda=0.01)
    for key in (
        "model_id", "model_revision", "dataset_id", "dataset_revision",
        "seed", "git_sha", "torch_version", "transformers_version",
        "mps_available", "forward_dtype", "stage", "damping_lambda",
    ):
        assert key in m, key
    assert m["model_revision"].startswith("7ae5576")

    artifact = tmp_path / "thing.parquet"
    artifact.write_bytes(b"x")
    p = save_manifest(artifact, m)
    assert p.name == "thing.parquet.provenance.json"
    assert json.loads(p.read_text())["stage"] == "test"

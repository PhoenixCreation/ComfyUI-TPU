"""Artifact manifest verification (spec section 3): missing artifacts, wrong
digests, and unpinned digests all fail startup; the hash script is
idempotent and pins every artifact."""

import hashlib
import json
import os

from comfy import tpu_profile

import deployment.hash_artifacts as hash_artifacts


def _write_manifest(tmp_path, artifacts):
    manifest = {"profile": "krea2-1920x1080", "artifacts": {}}
    for name, info in artifacts.items():
        manifest["artifacts"][name] = {"path": info[0], "dtype": "bf16", "sha256": info[1]}
    path = tmp_path / "model_manifest.json"
    path.write_text(json.dumps(manifest))
    return str(path)


def _artifact(tmp_path, name, content=b"payload"):
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_missing_artifact_fails(tmp_path):
    manifest = _write_manifest(tmp_path, {"krea2_turbo_bf16.safetensors": ("models/x.safetensors", "x" * 64)})
    results = tpu_profile.verify_artifacts(json.loads(open(manifest).read()))
    assert results[0]["status"] == "error"
    assert "missing" in results[0]["detail"].lower()


def test_wrong_digest_fails(tmp_path):
    f = _artifact(tmp_path, "x.safetensors")
    manifest = _write_manifest(tmp_path, {"krea2_turbo_bf16.safetensors": (str(f), "f" * 64)})
    results = tpu_profile.verify_artifacts(json.loads(open(manifest).read()))
    assert results[0]["status"] == "error"
    assert "sha256" in results[0]["detail"].lower()


def test_unpinned_digest_fails_with_guidance(tmp_path):
    f = _artifact(tmp_path, "x.safetensors")
    manifest = _write_manifest(tmp_path, {"krea2_turbo_bf16.safetensors": (str(f), "")})
    results = tpu_profile.verify_artifacts(json.loads(open(manifest).read()))
    assert results[0]["status"] == "error"
    assert "hash_artifacts.py" in results[0]["detail"]


def test_correct_digest_passes(tmp_path):
    f = _artifact(tmp_path, "x.safetensors", b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    manifest = _write_manifest(tmp_path, {"krea2_turbo_bf16.safetensors": (str(f), digest)})
    results = tpu_profile.verify_artifacts(json.loads(open(manifest).read()))
    assert results[0]["status"] == "ok"


def test_hash_artifacts_pins_all_and_is_idempotent(tmp_path, monkeypatch):
    for name in ("a.safetensors", "b.safetensors", "c.safetensors"):
        _artifact(tmp_path, name, content=name.encode())
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(json.dumps({
        "profile": "krea2-1920x1080",
        "artifacts": {
            "a.safetensors": {"path": "a.safetensors", "dtype": "bf16", "sha256": ""},
            "b.safetensors": {"path": "b.safetensors", "dtype": "bf16", "sha256": ""},
            "c.safetensors": {"path": "c.safetensors", "dtype": "bf16", "sha256": ""},
        },
    }))
    monkeypatch.setattr(hash_artifacts, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(hash_artifacts, "MANIFEST", str(manifest_path))
    assert hash_artifacts.main() == 0
    pinned = json.loads(manifest_path.read_text())
    for name, content in (("a.safetensors", b"a.safetensors"), ("b.safetensors", b"b.safetensors"),
                          ("c.safetensors", b"c.safetensors")):
        assert pinned["artifacts"][name]["sha256"] == hashlib.sha256(content).hexdigest()
    assert hash_artifacts.main() == 0


def test_hash_artifacts_reports_missing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "model_manifest.json"
    manifest_path.write_text(json.dumps({
        "profile": "krea2-1920x1080",
        "artifacts": {"a.safetensors": {"path": "missing.safetensors", "dtype": "bf16", "sha256": ""}},
    }))
    monkeypatch.setattr(hash_artifacts, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(hash_artifacts, "MANIFEST", str(manifest_path))
    assert hash_artifacts.main() == 1
    assert json.loads(manifest_path.read_text())["artifacts"]["a.safetensors"]["sha256"] == ""
"""Warm-up driver (spec section 14): artifact gating, output-node selection,
output redirection, and the readiness state machine, all tracked with fakes."""

import pytest

import comfy.accelerator
from comfy import tpu_profile
from comfy.cli_args import args


@pytest.fixture(autouse=True)
def _reset_readiness():
    r = tpu_profile.readiness
    r.last_error = ""
    r.fields = {}
    r.artifact_hashes = []
    r.mesh = ""
    r.state = "initializing"
    yield
    r.last_error = ""
    r.fields = {}
    r.artifact_hashes = []
    r.mesh = ""
    r.state = "initializing"


@pytest.fixture
def main(monkeypatch):
    """Import the production entrypoint with the XLA adapter faked so no
    torch_xla is loaded. main.py calls initialize_accelerator() at module
    import; the conftest stub satisfies the call without touching XLA."""
    monkeypatch.setattr(comfy.accelerator, "initialize_accelerator",
                        lambda: comfy.accelerator.get_accelerator())
    return __import__("main")


class FakeExecutor:
    def __init__(self, server=None):
        self.execute_calls = []
        self.success = True
        self.status_messages = []

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        self.execute_calls.append((prompt_id, list(execute_outputs)))


def test_warmup_fails_on_unpinned_manifest(monkeypatch, main):
    monkeypatch.setattr(tpu_profile, "verify_artifacts",
                        lambda m: [{"name": "krea2_turbo_bf16.safetensors", "status": "error",
                                    "detail": "sha256 not pinned in manifest; run deployment/hash_artifacts.py after placing artifacts"}])
    main.run_tpu_warmup(None)
    assert tpu_profile.readiness.state == "failed"
    assert "hash_artifacts.py" in tpu_profile.readiness.last_error


def test_warmup_fails_on_missing_manifest(monkeypatch, main):
    monkeypatch.setattr(tpu_profile, "load_manifest", lambda: {})
    main.run_tpu_warmup(None)
    assert tpu_profile.readiness.state == "failed"


def test_warmup_executes_output_nodes_and_redirects_output(monkeypatch, main, tmp_path):
    fake = FakeExecutor()
    monkeypatch.setattr(tpu_profile, "verify_artifacts",
                        lambda m: [{"name": "krea2_turbo_bf16.safetensors", "status": "ok", "detail": ""}])
    monkeypatch.setattr(main.execution, "PromptExecutor", lambda server: fake)

    import folder_paths
    calls = {}
    monkeypatch.setattr(folder_paths, "set_output_directory", lambda path: calls.setdefault("dir", path))
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: "/fake/original")
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    main.run_tpu_warmup(None)

    assert fake.execute_calls, "executor.execute was never called"
    prompt_id, outputs = fake.execute_calls[0]
    assert prompt_id == "tpu-warmup"
    assert set(outputs) == {"5", "14"}
    assert "warmup_output" in calls["dir"]
    assert tpu_profile.readiness.fields.get("compile_counters_delta") is not None
    assert tpu_profile.readiness.state == "ready"


def test_warmup_fails_when_execution_fails(monkeypatch, main, tmp_path):
    class FailingExecutor(FakeExecutor):
        def __init__(self, server=None):
            super().__init__(server)
            self.success = False
            self.status_messages = [{"message": "boom"}]

    monkeypatch.setattr(tpu_profile, "verify_artifacts",
                        lambda m: [{"name": "krea2_turbo_bf16.safetensors", "status": "ok", "detail": ""}])
    monkeypatch.setattr(main.execution, "PromptExecutor", lambda server: FailingExecutor())

    import folder_paths
    monkeypatch.setattr(folder_paths, "set_output_directory", lambda path: None)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: "/fake/original")
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path))

    main.run_tpu_warmup(None)
    assert tpu_profile.readiness.state == "failed"
    assert "boom" in tpu_profile.readiness.last_error


def test_warmup_skips_compile_when_disabled(monkeypatch, main):
    saved = args.tpu_warmup
    args.tpu_warmup = False
    try:
        monkeypatch.setattr(tpu_profile, "verify_artifacts",
                            lambda m: [{"name": "krea2_turbo_bf16.safetensors", "status": "ok", "detail": ""}])
        main.run_tpu_warmup(None)
        assert tpu_profile.readiness.state == "ready"
        assert "compile_counters_delta" not in tpu_profile.readiness.fields
    finally:
        args.tpu_warmup = saved


def test_warmup_workflow_has_output_nodes(canonical_workflow):
    outputs = [nid for nid, node in canonical_workflow.items()
               if node.get("class_type") in ("SaveImage", "PreviewImage")]
    assert set(outputs) == {"5", "14"}
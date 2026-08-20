"""Stage tracker and per-request instrumentation record (spec section 15)."""

import logging

import comfy.accelerator


def test_stage_timer_accumulates(tpu_mode):
    tracker = comfy.accelerator.StageTracker("test-1", profile="krea2-1920x1080")
    comfy.accelerator.set_current_tracker(tracker)
    try:
        with comfy.accelerator.stage_timer("denoising"):
            pass
        with comfy.accelerator.stage_timer("denoising"):
            pass
        with comfy.accelerator.stage_timer("vae"):
            pass
    finally:
        comfy.accelerator.set_current_tracker(None)
    assert tracker.stages["denoising"] > 0
    assert tracker.stages["vae"] > 0


def test_emit_writes_structured_record(tpu_mode, caplog):
    tracker = comfy.accelerator.StageTracker("test-2", profile="krea2-1920x1080")
    tracker.begin_interval()
    tracker.record("compile_counters_delta", {"MarkStep": 9})
    with caplog.at_level(logging.INFO):
        tracker.emit("ok")
    assert caplog.records[-1].message.startswith("tpu_request")
    record = caplog.records[-1].getMessage().split(" ", 1)[1]
    assert isinstance(record, str)
    assert "test-2" in record


def test_to_log_record_fields(tpu_mode):
    tracker = comfy.accelerator.StageTracker("test-3", profile="krea2-1920x1080")
    tracker.begin_interval()
    tracker.begin("tokenization")
    tracker.end("tokenization")
    tracker.finalize()
    record = tracker.to_log_record("ok")
    assert record["event"] == "tpu_request"
    assert record["prompt_id"] == "test-3"
    assert record["profile"] == "krea2-1920x1080"
    assert record["outcome"] == "ok"
    assert "tokenization" in record["durations_ms"]
    assert record["execution_interval_ms"] >= 0


def test_no_tracker_no_crash(tpu_mode):
    comfy.accelerator.set_current_tracker(None)
    with comfy.accelerator.stage_timer("denoising"):
        pass
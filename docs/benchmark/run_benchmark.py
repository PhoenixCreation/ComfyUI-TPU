#!/usr/bin/env python3
"""
Benchmark harness for Krea2 1920x1080 on TPU v5e-8.
Implements cold vs warm scenarios with repeated trials to exclude flukes.

- Cold: fresh process + fresh TPU_CACHE_DIR + --tpu-warmup (server compiles at startup)
- Warm: steady-state requests in same process after warmup

For each cold trial:
  launch server -> wait ready -> record warmup_timestamps -> run N warm requests -> kill

Metrics per request:
  wall_e2e_s, execution_interval_ms, durations_ms per stage, compile counters, output verification, memory

Produces JSON per trial under /tmp/benchmark_results/*.json and aggregated stats printed.
"""
import os
import sys
import json
import time
import shutil
import signal
import subprocess
import urllib.request
import urllib.error
import statistics
import pathlib
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / "workflows" / "Krea2-turbo-tpu.json"
OUTPUT_DIR = REPO_ROOT / "output"
RESULT_DIR = pathlib.Path("/tmp/benchmark_results")
LOG_DIR = pathlib.Path("/tmp/benchmark_logs")

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8188
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

# Benchmark config
COLD_TRIALS = 3
WARM_PER_COLD = 5  # warm requests after each cold startup
NO_WARMUP_TRIALS = 2
WARM_PER_NO_WARMUP = 3  # includes first cold request + subsequent warms

# Timeouts (seconds)
READY_TIMEOUT = 400  # warmup compile can be ~105s but allow more
REQUEST_TIMEOUT = 300
POLL_INTERVAL = 2.0

TPU_ENV = {
    "PJRT_DEVICE": "TPU",
    "TPU_SKIP_MDS_QUERY": "1",
    "TPU_ACCELERATOR_TYPE": "v5litepod-8",
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,4,1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_WORKER_ID": "0",
    "TPU_WORKER_HOSTNAMES": "localhost",
}


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_workflow():
    with open(WORKFLOW_PATH) as f:
        return json.load(f)


def http_get(path, timeout=10):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_post_prompt(prompt, timeout=10):
    url = f"{BASE_URL}/prompt"
    data = json.dumps({"prompt": prompt}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def poll_ready(proc, log_file, timeout=READY_TIMEOUT):
    """Poll /tpu/status until ready or failed or timeout."""
    start = time.monotonic()
    last_state = None
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            # process died
            try:
                with open(log_file) as f:
                    tail = "".join(f.readlines()[-100:])
            except:
                tail = ""
            raise RuntimeError(f"Server process died (exit {proc.returncode}). Tail:\n{tail}")
        try:
            snap = http_get("/tpu/status", timeout=5)
            state = snap.get("state")
            if state != last_state:
                print(f"  [poll_ready] state={state} after {time.monotonic()-start:.1f}s")
                last_state = state
            if state == "ready":
                return snap
            if state == "failed":
                raise RuntimeError(f"TPU readiness failed: {snap.get('last_error')} snapshot={snap}")
        except urllib.error.URLError as e:
            # server not yet up
            pass
        except json.JSONDecodeError:
            pass
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Timed out waiting for ready after {timeout}s")


def wait_for_history(prompt_id, timeout=REQUEST_TIMEOUT):
    """Poll /history/<prompt_id> until success/error."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            hist = http_get(f"/history/{prompt_id}", timeout=5)
            # hist is dict keyed by prompt_id
            if prompt_id in hist:
                entry = hist[prompt_id]
                # check status?
                # promptQueue returns history with status_str
                # but some versions store differently
                # Look for outputs
                if "outputs" in entry and entry["outputs"]:
                    return entry
                # also check if execution failed: look for status str
                # history endpoint may return {"status": {"status_str": "success"}}
                # But for simplicity, if entry has outputs, consider success
                # Check server's execution status via presence of outputs
                status = entry.get("status", {})
                if isinstance(status, dict) and status.get("status_str") == "error":
                    raise RuntimeError(f"Execution failed for {prompt_id}: {status}")
            # Also check general /history
        except urllib.error.URLError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"History poll timeout for {prompt_id} after {timeout}s")


def kill_proc(proc, log_file):
    if proc.poll() is None:
        print(f"  killing server pid={proc.pid}")
        try:
            # Try graceful SIGTERM
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("  SIGTERM timeout, SIGKILL")
                proc.kill()
                proc.wait(timeout=10)
        except Exception as e:
            print(f"  kill error: {e}")
    # also ensure no orphan python main.py --tpu
    try:
        subprocess.run(["pkill", "-f", "python main.py.*--tpu"], timeout=5)
    except:
        pass
    time.sleep(3)  # let TPU devices release
    # dump log tail for debugging if needed
    try:
        with open(log_file) as f:
            lines = f.readlines()
            if lines:
                print(f"  log tail {len(lines)} lines, last 5:")
                for l in lines[-5:]:
                    print(f"    {l.rstrip()}")
    except:
        pass


def launch_server(cache_dir, with_warmup=True):
    """Launch ComfyUI TPU server, return proc, log_file."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous cache contents for cold? But we already use fresh dir per trial
    log_file = LOG_DIR / f"server_{cache_dir.name}.log"
    env = os.environ.copy()
    env.update(TPU_ENV)
    # Ensure unset legacy vars
    env.pop("TPU_PROCESS_ADDRESSES", None)
    env.pop("XRT_TPU_CONFIG", None)
    env["XLA_PERSISTENT_CACHE_PATH"] = str(cache_dir / "executables" / "placeholder")  # will be overwritten by XlaAccelerator fingerprint
    # Unset to avoid parent contamination? XlaAccelerator sets it anyway
    # But ensure PJRT correctly
    cmd = [
        sys.executable, "main.py",
        "--tpu",
        "--tpu-cache-dir", str(cache_dir),
        "--tpu-profile", "krea2-1920x1080",
        "--listen", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--disable-auto-launch",
        "--disable-metadata",
        "--input-directory", str(REPO_ROOT / "input"),
        "--output-directory", str(REPO_ROOT / "output"),
    ]
    if with_warmup:
        cmd.append("--tpu-warmup")
    else:
        cmd.append("--no-tpu-warmup")

    print(f"\n=== Launching server cache={cache_dir} warmup={with_warmup} ===")
    print(f"cmd: {' '.join(cmd)}")
    print(f"log: {log_file}")
    # Ensure previous server not holding port
    try:
        subprocess.run(["fuser", "-k", f"{SERVER_PORT}/tcp"], timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    time.sleep(2)
    log_fd = open(log_file, "w")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_fd, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    # Give a moment to see if immediate crash
    time.sleep(3)
    if proc.poll() is not None:
        log_fd.close()
        with open(log_file) as f:
            tail = "".join(f.readlines()[-100:])
        raise RuntimeError(f"Server exited immediately {proc.returncode} tail:\n{tail}")
    return proc, log_file, log_fd


def verify_output(entry):
    """Verify output file exists, is RGB 1920x1080, return path+size. Prefer output/ over temp/."""
    from PIL import Image
    outputs = entry.get("outputs", {})
    candidates = []
    for node_id, out in outputs.items():
        for k, v in out.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "filename" in item:
                        filename = item["filename"]
                        subfolder = item.get("subfolder", "")
                        typ = item.get("type", "output")
                        base = REPO_ROOT / "output" if typ == "output" else REPO_ROOT / "temp"
                        if subfolder:
                            path = base / subfolder / filename
                        else:
                            path = base / filename
                        candidates.append((typ, path, item))
    # Prefer output candidates first
    candidates.sort(key=lambda x: 0 if x[0] == "output" else 1)
    for typ, path, item in candidates:
        if path.exists():
            try:
                with Image.open(path) as im:
                    w, h = im.size
                    mode = im.mode
                    return {"path": str(path), "width": w, "height": h, "mode": mode, "valid": w==1920 and h==1080 and mode=="RGB", "type": typ, "filename": item.get("filename")}
            except Exception as e:
                return {"path": str(path), "error": str(e), "valid": False, "type": typ}
    return {"path": None, "valid": False, "error": "no output file found", "candidates": len(candidates)}


def _strip_ansi(s):
    import re
    ansi = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi.sub('', s)

def parse_tpu_request_logs(log_file, prompt_id, retries=5):
    """Try to find tpu_request log line for prompt_id. Strip ANSI and handle buffering."""
    import re
    for attempt in range(retries):
        try:
            with open(log_file, errors="ignore") as f:
                content = _strip_ansi(f.read())
                # Find lines containing tpu_request and prompt_id
                for line in content.splitlines():
                    if "tpu_request" in line and prompt_id in line:
                        idx = line.find("{")
                        if idx != -1:
                            # Extract JSON; find matching braces (payload is single line)
                            # Use raw substring from first { to last } on line
                            json_str = line[idx:]
                            # Handle trailing ANSI or extra chars: find last }
                            last = json_str.rfind("}")
                            if last != -1:
                                json_str = json_str[:last+1]
                            try:
                                payload = json.loads(json_str)
                                return payload
                            except Exception:
                                # Try more robust: extract via regex
                                m = re.search(r'\{.*\}', line)
                                if m:
                                    try:
                                        return json.loads(m.group(0))
                                    except:
                                        pass
        except Exception:
            pass
        # Fallback: scan entire file for JSON objects containing prompt_id
        try:
            with open(log_file, errors="ignore") as f:
                content = _strip_ansi(f.read())
                # Find all JSON objects that look like tpu_request
                # Use iterative brace matching instead of non-greedy regex
                for m in re.finditer(r'"event"\s*:\s*"tpu_request".*?"prompt_id"\s*:\s*"%s"' % re.escape(prompt_id), content, re.DOTALL):
                    # Try to extract surrounding braces
                    start = content.rfind("{", 0, m.start())
                    # Find matching end by counting braces
                    depth = 0
                    end = -1
                    for i in range(start, len(content)):
                        if content[i] == "{":
                            depth += 1
                        elif content[i] == "}":
                            depth -= 1
                            if depth == 0:
                                end = i
                                break
                    if start != -1 and end != -1:
                        try:
                            return json.loads(content[start:end+1])
                        except:
                            continue
        except:
            pass
        if attempt < retries - 1:
            time.sleep(1)
    return None


def get_system_info():
    info = {}
    try:
        os.environ["TPU_SKIP_MDS_QUERY"] = "1"
        import torch, torch_xla
        info["torch"] = torch.__version__
        info["torch_xla"] = torch_xla.__version__
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import libtpu
        info["libtpu"] = libtpu.__version__ if hasattr(libtpu, "__version__") else "0.0.17"
    except:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemTotal" in line or "MemAvailable" in line:
                    info[line.split(":")[0]] = line.strip()
    except:
        pass
    try:
        import psutil
        info["cpu_count"] = psutil.cpu_count()
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 1)
        info["ram_available_gb"] = round(vm.available / (1024**3), 1)
    except:
        pass
    return info


def run_one_cold_trial(trial_idx, with_warmup=True):
    cache_dir = f"/tmp/tpu-cache-krea2-benchmark-{'warmup' if with_warmup else 'nowarmup'}-trial{trial_idx}"
    # ensure fresh
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    proc, log_file, log_fd = launch_server(cache_dir, with_warmup=with_warmup)
    trial_result = {
        "trial_idx": trial_idx,
        "with_warmup": with_warmup,
        "cache_dir": cache_dir,
        "log_file": str(log_file),
        "start_time": datetime.utcnow().isoformat() + "Z",
    }
    try:
        # measure startup to ready
        startup_start = time.monotonic()
        snap = poll_ready(proc, log_file)
        ready_elapsed = time.monotonic() - startup_start
        trial_result["ready_snapshot"] = snap
        trial_result["startup_to_ready_s"] = round(ready_elapsed, 2)

        # parse warmup timestamps
        ts = snap.get("warmup_timestamps", {})
        if ts:
            # values are monotonic floats
            keys = ["initializing", "loading", "compiling", "ready"]
            diffs = {}
            for i in range(1, len(keys)):
                if keys[i] in ts and keys[i-1] in ts and ts[keys[i]] is not None and ts[keys[i-1]] is not None:
                    diffs[f"{keys[i-1]}->{keys[i]}"] = round(ts[keys[i]] - ts[keys[i-1]], 2)
            # total warmup = ready - initializing
            if "initializing" in ts and "ready" in ts and ts["ready"] and ts["initializing"]:
                diffs["total_initializing->ready"] = round(ts["ready"] - ts["initializing"], 2)
            trial_result["warmup_diffs_s"] = diffs
            print(f"  warmup diffs: {diffs}")

        # also capture compile counters delta if present
        if "fields" in snap and "compile_counters_delta" in snap["fields"]:
            trial_result["warmup_compile_delta"] = snap["fields"]["compile_counters_delta"]

        # memory after warmup
        try:
            import psutil
            vm = psutil.virtual_memory()
            proc_mem = psutil.Process(proc.pid).memory_info()
            trial_result["mem_after_warmup"] = {
                "host_available_gb": round(vm.available / (1024**3), 2),
                "host_used_gb": round(vm.used / (1024**3), 2),
                "proc_rss_gb": round(proc_mem.rss / (1024**3), 2),
            }
        except Exception as e:
            trial_result["mem_after_warmup_error"] = str(e)

        # Now run warm requests
        workflow = load_workflow()
        # For determinism, vary seed per trial/request
        base_seed = 1010070471918926
        requests = []
        n_warm = WARM_PER_COLD if with_warmup else WARM_PER_NO_WARMUP
        for req_idx in range(n_warm):
            # vary prompt text slightly to ensure not cached? But cache is RAM pressure based on inputs?
            # For TPU, we want to measure steady-state after compilation, so same shapes but different seed/text should still reuse compilation.
            # We'll vary seed and prompt text deterministically.
            prompt = json.loads(json.dumps(workflow))  # deep copy
            # vary seed
            new_seed = base_seed + trial_idx * 100 + req_idx
            if "2" in prompt:
                prompt["2"]["inputs"]["seed"] = new_seed
            # vary prompt text for even indices to test conditioning but same token length? Keep same for now to avoid tokenizer variation.
            # We'll keep text same to ensure tokenizer fixed.

            # Submit
            print(f"  [trial {trial_idx}][req {req_idx}] submitting seed={new_seed}")
            e2e_start = time.monotonic()
            # For no_warmup first request, this will include compilation
            resp = http_post_prompt(prompt)
            prompt_id = resp.get("prompt_id") or resp.get("promptID") or list(resp.values())[0] if isinstance(resp, dict) and len(resp)==1 else resp.get("prompt_id")
            if not prompt_id:
                # response format: {"prompt_id": "...", "number": ..., "node_errors": {}}
                prompt_id = resp.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"No prompt_id in response {resp}")
            print(f"    prompt_id={prompt_id} submit took {time.monotonic()-e2e_start:.2f}s, waiting history...")
            hist_entry = wait_for_history(prompt_id)
            e2e_elapsed = time.monotonic() - e2e_start
            print(f"    completed e2e {e2e_elapsed:.2f}s")

            # verify output
            verify = verify_output(hist_entry)
            print(f"    verify: {verify}")

            # parse tpu_request log for this prompt_id
            # wait a bit for log flush (give XLA time to emit)
            time.sleep(2)
            tracker = parse_tpu_request_logs(log_file, prompt_id, retries=3)
            # also try to get execution_interval from tracker
            req_record = {
                "req_idx": req_idx,
                "seed": new_seed,
                "prompt_id": prompt_id,
                "e2e_s": round(e2e_elapsed, 3),
                "verify": verify,
                "history": hist_entry,
                "tracker": tracker,
            }
            if tracker:
                req_record["execution_interval_ms"] = tracker.get("execution_interval_ms")
                req_record["durations_ms"] = tracker.get("durations_ms")
                req_record["compile_counters_before"] = tracker.get("compile_counters_before")
                req_record["compile_counters_after"] = tracker.get("compile_counters_after")
                req_record["memory"] = tracker.get("memory")
                print(f"    tracker interval {tracker.get('execution_interval_ms')} ms durations {tracker.get('durations_ms')}")
            else:
                print(f"    no tracker found for {prompt_id}")

            requests.append(req_record)
            # brief pause between requests to let system settle
            time.sleep(2)

        trial_result["requests"] = requests
        # aggregate warm stats for this trial (exclude first if no_warmup? but we keep all)
        e2es = [r["e2e_s"] for r in requests]
        intervals = [r["execution_interval_ms"] for r in requests if r.get("execution_interval_ms") is not None]
        trial_result["stats_e2e"] = {
            "mean": round(statistics.mean(e2es), 3) if e2es else None,
            "median": round(statistics.median(e2es), 3) if e2es else None,
            "stdev": round(statistics.stdev(e2es), 3) if len(e2es) > 1 else 0,
            "min": round(min(e2es), 3) if e2es else None,
            "max": round(max(e2es), 3) if e2es else None,
        }
        if intervals:
            trial_result["stats_interval_ms"] = {
                "mean": round(statistics.mean(intervals), 2),
                "median": round(statistics.median(intervals), 2),
                "stdev": round(statistics.stdev(intervals), 2) if len(intervals) > 1 else 0,
                "min": round(min(intervals), 2),
                "max": round(max(intervals), 2),
            }

        # final memory
        try:
            import psutil
            vm = psutil.virtual_memory()
            proc_mem = psutil.Process(proc.pid).memory_info()
            trial_result["mem_after_requests"] = {
                "host_available_gb": round(vm.available / (1024**3), 2),
                "proc_rss_gb": round(proc_mem.rss / (1024**3), 2),
            }
        except:
            pass

    finally:
        # close log fd
        try:
            log_fd.close()
        except:
            pass
        kill_proc(proc, log_file)
        # collect cache profile if exists
        try:
            cache_profile = pathlib.Path(cache_dir) / "cache_profile.json"
            if cache_profile.exists():
                with open(cache_profile) as f:
                    trial_result["cache_profile"] = json.load(f)
            sharding_report = pathlib.Path(cache_dir) / "sharding_report.json"
            if sharding_report.exists():
                with open(sharding_report) as f:
                    trial_result["sharding_report"] = json.load(f)
            # list executables dir
            exec_dir = pathlib.Path(cache_dir) / "executables"
            if exec_dir.exists():
                trial_result["cache_executables"] = [str(p) for p in exec_dir.iterdir()]
        except Exception as e:
            trial_result["cache_collect_error"] = str(e)

    # Save trial json
    out_path = RESULT_DIR / f"trial_{'warmup' if with_warmup else 'nowarmup'}_{trial_idx}.json"
    with open(out_path, "w") as f:
        json.dump(trial_result, f, indent=2)
    print(f"  trial {trial_idx} saved to {out_path}")
    return trial_result


def main():
    ensure_dirs()
    print("=== Krea2 1920x1080 TPU Benchmark ===")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Workflow: {WORKFLOW_PATH}")
    print(f"COLD_TRIALS={COLD_TRIALS} WARM_PER_COLD={WARM_PER_COLD} NO_WARMUP_TRIALS={NO_WARMUP_TRIALS}")
    print(f"System info: {get_system_info()}")

    # Clear output before start? Keep but note
    all_trials = []
    # Warmup cold trials
    for i in range(1, COLD_TRIALS + 1):
        print(f"\n### COLD TRIAL {i}/{COLD_TRIALS} (with warmup) ###")
        res = run_one_cold_trial(i, with_warmup=True)
        all_trials.append(res)
        # pause between trials
        time.sleep(5)

    # No-warmup cold trials (first request pays compile)
    for i in range(1, NO_WARMUP_TRIALS + 1):
        idx = COLD_TRIALS + i
        print(f"\n### NO-WARMUP TRIAL {i}/{NO_WARMUP_TRIALS} (without warmup) ###")
        res = run_one_cold_trial(idx, with_warmup=False)
        all_trials.append(res)
        time.sleep(5)

    # Aggregate overall warm stats
    warm_e2es = []
    warm_intervals = []
    cold_startups = []
    for t in all_trials:
        if t.get("with_warmup"):
            cold_startups.append(t.get("startup_to_ready_s"))
            # warm requests after warmup: consider all requests as warm
            for r in t.get("requests", []):
                warm_e2es.append(r["e2e_s"])
                if r.get("execution_interval_ms"):
                    warm_intervals.append(r["execution_interval_ms"])
        else:
            # for no-warmup, first request is cold, remaining are warm
            reqs = t.get("requests", [])
            if reqs:
                # first is cold compilation
                cold_startups.append(reqs[0]["e2e_s"])  # treat first e2e as cold
                for r in reqs[1:]:
                    warm_e2es.append(r["e2e_s"])
                    if r.get("execution_interval_ms"):
                        warm_intervals.append(r["execution_interval_ms"])
            # also startup without warmup is fast
            cold_startups.append(t.get("startup_to_ready_s"))

    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "COLD_TRIALS": COLD_TRIALS,
            "WARM_PER_COLD": WARM_PER_COLD,
            "NO_WARMUP_TRIALS": NO_WARMUP_TRIALS,
            "WARM_PER_NO_WARMUP": WARM_PER_NO_WARMUP,
            "profile": "krea2-1920x1080",
            "resolution": "1920x1080",
            "steps": 8,
            "sampler": "er_sde/simple",
            "cfg": 1.0,
        },
        "system": get_system_info(),
        "cold_startups_s": cold_startups,
        "warm_e2e_s": warm_e2es,
        "warm_intervals_ms": warm_intervals,
    }
    if cold_startups:
        summary["stats_cold_startup"] = {
            "mean": round(statistics.mean(cold_startups), 2),
            "median": round(statistics.median(cold_startups), 2),
            "stdev": round(statistics.stdev(cold_startups), 2) if len(cold_startups)>1 else 0,
            "min": round(min(cold_startups), 2),
            "max": round(max(cold_startups), 2),
            "count": len(cold_startups),
        }
    if warm_e2es:
        summary["stats_warm_e2e"] = {
            "mean": round(statistics.mean(warm_e2es), 3),
            "median": round(statistics.median(warm_e2es), 3),
            "stdev": round(statistics.stdev(warm_e2es), 3) if len(warm_e2es)>1 else 0,
            "min": round(min(warm_e2es), 3),
            "max": round(max(warm_e2es), 3),
            "p90": round(sorted(warm_e2es)[int(0.9*len(warm_e2es))-1], 3) if len(warm_e2es)>=10 else round(sorted(warm_e2es)[-1], 3),
            "count": len(warm_e2es),
        }
    if warm_intervals:
        summary["stats_warm_interval_ms"] = {
            "mean": round(statistics.mean(warm_intervals), 2),
            "median": round(statistics.median(warm_intervals), 2),
            "stdev": round(statistics.stdev(warm_intervals), 2) if len(warm_intervals)>1 else 0,
            "min": round(min(warm_intervals), 2),
            "max": round(max(warm_intervals), 2),
            "p90": round(sorted(warm_intervals)[int(0.9*len(warm_intervals))-1], 2) if len(warm_intervals)>=10 else round(sorted(warm_intervals)[-1], 2),
            "count": len(warm_intervals),
        }

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    with open(RESULT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(RESULT_DIR / "all_trials.json", "w") as f:
        json.dump(all_trials, f, indent=2)

    print(f"\nResults saved to {RESULT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

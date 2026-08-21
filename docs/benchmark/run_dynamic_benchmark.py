#!/usr/bin/env python3
"""
Dynamic Krea2 benchmark: 1024x1024, 1080x1920, 1152x896, 1280x720 etc.
Measures cold (first at size, compiles) vs warm (reuses) to show
compilation stays for next execution. Repeats to exclude flukes.
"""
import os, sys, json, time, subprocess, pathlib, urllib.request, urllib.error, re, statistics
from datetime import datetime
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT/"workflows/Krea2-turbo-tpu.json"
BASE_URL = "http://127.0.0.1:8188"
TPU_ENV = {
    "PJRT_DEVICE": "TPU",
    "TPU_SKIP_MDS_QUERY": "1",
    "TPU_ACCELERATOR_TYPE": "v5litepod-8",
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,4,1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_WORKER_ID": "0",
    "TPU_WORKER_HOSTNAMES": "localhost",
}
# Sizes: 1024x1024 (1M), 1080x1920 (2M portrait), 1152x896 (1M), 1280x720 (0.92M), 1024x768 (0.79M) - all valid per tpu_profile
SIZES = [
    (1024, 1024),
    (1080, 1920),
    (1152, 896),
    (1280, 720),
    (1024, 768),
]

RESULT_DIR = pathlib.Path("/tmp/benchmark_dynamic_results")
LOG_DIR = pathlib.Path("/tmp/benchmark_dynamic_logs")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

def http_get(p, timeout=10):
    with urllib.request.urlopen(BASE_URL+p, timeout=timeout) as r:
        return json.loads(r.read().decode())
def http_post(prompt, timeout=10):
    data=json.dumps({"prompt":prompt}).encode()
    req=urllib.request.Request(BASE_URL+"/prompt", data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())
def wait_history(pid, timeout=300):
    s=time.monotonic()
    while time.monotonic()-s<timeout:
        try:
            h=http_get(f"/history/{pid}", timeout=10)
            if pid in h and h[pid].get("outputs"):
                return h[pid]
            if pid in h and h[pid].get("status",{}).get("status_str")=="error":
                return h[pid]
        except: pass
        time.sleep(1)
    raise TimeoutError(pid)
def strip_ansi(s):
    return re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]').sub('', s)
def parse_tracker(log_path, pid, retries=5):
    for _ in range(retries):
        try:
            with open(log_path, errors="ignore") as f:
                c=strip_ansi(f.read())
                for line in c.splitlines():
                    if "tpu_request" in line and pid in line:
                        idx=line.find("{")
                        if idx!=-1:
                            j=line[idx:line.rfind("}")+1]
                            try: return json.loads(j)
                            except: pass
        except: pass
        time.sleep(1)
    # fallback search all
    try:
        with open(log_path, errors="ignore") as f:
            c=strip_ansi(f.read())
            for m in re.finditer(r'"event"\s*:\s*"tpu_request".*?"prompt_id"\s*:\s*"%s"' % re.escape(pid), c, re.DOTALL):
                start=c.rfind("{",0,m.start())
                depth=0
                end=-1
                for i in range(start,len(c)):
                    if c[i]=="{": depth+=1
                    elif c[i]=="}":
                        depth-=1
                        if depth==0:
                            end=i
                            break
                if start!=-1 and end!=-1:
                    try: return json.loads(c[start:end+1])
                    except: continue
    except: pass
    return None

def launch(profile="krea2"):
    cache=f"/tmp/tpu-cache-krea2-dynamic-bench-{profile}"
    if os.path.exists(cache):
        import shutil; shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(cache, exist_ok=True)
    log=LOG_DIR/f"server_{profile}.log"
    env=os.environ.copy()
    env.update(TPU_ENV)
    env.pop("TPU_PROCESS_ADDRESSES",None); env.pop("XRT_TPU_CONFIG",None)
    cmd=[sys.executable,"main.py","--tpu","--tpu-cache-dir",cache,"--tpu-profile",profile,"--listen","127.0.0.1","--port","8188","--disable-auto-launch","--disable-metadata","--input-directory",str(REPO_ROOT/"input"),"--output-directory",str(REPO_ROOT/"output"),"--tpu-warmup"]
    print(f"\n=== Launch profile={profile} cache={cache} log={log} ===")
    print(" ".join(cmd))
    try: subprocess.run(["fuser","-k","8188/tcp"], timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except: pass
    time.sleep(2)
    lf=open(log,"w")
    proc=subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    time.sleep(3)
    if proc.poll() is not None:
        lf.close()
        with open(log) as f: print(f.read()[-5000:])
        raise RuntimeError(f"Server died {proc.returncode}")
    return proc, log, lf, cache

def snapshot_ready(proc, log, timeout=600):
    s=time.monotonic()
    last=None
    while time.monotonic()-s<timeout:
        if proc.poll() is not None:
            with open(log) as f: print(f.read()[-5000:])
            raise RuntimeError(f"died {proc.returncode}")
        try:
            snap=http_get("/tpu/status", timeout=10)
            st=snap.get("state")
            if st!=last:
                print(f"  state {st} after {time.monotonic()-s:.1f}s")
                last=st
            if st=="ready":
                return snap
            if st=="failed":
                raise RuntimeError(f"failed {snap.get('last_error')}")
        except urllib.error.URLError: pass
        time.sleep(2)
    raise TimeoutError("ready timeout")

def benchmark_profile(profile):
    proc, log, lf, cache = launch(profile)
    try:
        snap=snapshot_ready(proc, log)
        print(f"ready profile {snap.get('profile')} warmup {snap.get('warmup_timestamps')}")
        # capture startup diffs
        ts=snap.get("warmup_timestamps",{})
        diffs={}
        for a,b in [("initializing","loading"),("loading","compiling"),("compiling","ready")]:
            if a in ts and b in ts and ts[a] and ts[b]:
                diffs[f"{a}->{b}"]=round(ts[b]-ts[a],2)
        if "initializing" in ts and "ready" in ts and ts["initializing"] and ts["ready"]:
            diffs["total"]=round(ts["ready"]-ts["initializing"],2)
        print("warmup diffs", diffs)

        with open(WORKFLOW) as f: base=json.load(f)
        from PIL import Image

        all_results=[]
        # For each size, do 1 cold (first) + 2 warm (same size, different seeds, same text to test cached path)
        # Then also 1 warm with varying text (uncached) to show W2 vs W3
        base_seed=1010070471918926
        size_results={}
        for idx,(w,h) in enumerate(SIZES):
            print(f"\n--- Size {w}x{h} ---")
            res_list=[]
            # Use same text for first two to test cached path, third with varying text
            texts=[
                "A landscape painting by Albert Bierstadt.",
                "A landscape painting by Albert Bierstadt.",  # same -> cached
                f"A landscape painting by Albert Bierstadt. Variation {w}x{h} {idx}", # varying -> uncached
            ]
            for rep in range(3):
                p=json.loads(json.dumps(base))
                p["10"]["inputs"]["width"]=w
                p["10"]["inputs"]["height"]=h
                p["2"]["inputs"]["seed"]=base_seed + idx*100 + rep
                p["6"]["inputs"]["text"]=texts[rep]
                e2e_start=time.monotonic()
                resp=http_post(p)
                pid=resp["prompt_id"]
                print(f"  [{w}x{h} rep{rep} {'cached' if rep==1 else 'uncached' if rep==2 else 'cold'}] pid {pid} seed {p['2']['inputs']['seed']}")
                hist=wait_history(pid)
                e2e=time.monotonic()-e2e_start
                tracker=parse_tracker(log, pid)
                # verify
                outputs=hist.get("outputs",{})
                valid=False
                path=None
                for nid,out in outputs.items():
                    for k,v in out.items():
                        if isinstance(v,list):
                            for item in v:
                                if isinstance(item,dict) and "filename" in item:
                                    pth=REPO_ROOT/"output"/item["filename"] if item.get("type","output")=="output" else REPO_ROOT/"temp"/item["filename"]
                                    if pth.exists():
                                        try:
                                            with Image.open(pth) as im:
                                                if im.size==(w,h) and im.mode=="RGB":
                                                    valid=True
                                                    path=str(pth)
                                        except: pass
                interval=tracker.get("execution_interval_ms") if tracker else None
                durations=tracker.get("durations_ms") if tracker else None
                print(f"    e2e {e2e:.2f}s interval {interval} durations {durations} valid {valid}")
                rec={"w":w,"h":h,"rep":rep,"pid":pid,"e2e":round(e2e,2),"interval":interval,"durations":durations,"valid":valid,"path":path,"text":texts[rep]}
                if tracker:
                    rec["compile_before"]=tracker.get("compile_counters_before",{}).get("UncachedCompile")
                    rec["compile_after"]=tracker.get("compile_counters_after",{}).get("UncachedCompile")
                res_list.append(rec)
                all_results.append(rec)
                time.sleep(2)
            size_results[f"{w}x{h}"]=res_list

        # Revisiting first size to show cache stays across sizes
        print(f"\n--- Revisit 1024x1024 to show cache persistence ---")
        w,h=1024,1024
        p=json.loads(json.dumps(base))
        p["10"]["inputs"]["width"]=w; p["10"]["inputs"]["height"]=h
        p["2"]["inputs"]["seed"]=9999999
        p["6"]["inputs"]["text"]="A landscape painting by Albert Bierstadt."
        e2e_start=time.monotonic()
        resp=http_post(p)
        pid=resp["prompt_id"]
        hist=wait_history(pid)
        e2e=time.monotonic()-e2e_start
        tracker=parse_tracker(log,pid)
        print(f"  revisit 1024x1024 e2e {e2e:.2f}s interval {tracker.get('execution_interval_ms') if tracker else None} durations {tracker.get('durations_ms') if tracker else None}")

        # Save
        out=RESULT_DIR/f"dynamic_{profile}.json"
        with open(out,"w") as f:
            json.dump({"profile":profile,"warmup_diffs":diffs,"ready_snapshot":snap,"results":all_results,"size_results":size_results, "revisit_e2e":round(e2e,2)}, f, indent=2)
        print(f"saved {out}")

        # summary stats
        for key, lst in size_results.items():
            cold_e2e=lst[0]["e2e"]
            warm_cached_e2e=lst[1]["e2e"]
            warm_uncached_e2e=lst[2]["e2e"]
            print(f"{key}: cold {cold_e2e}s -> warm cached {warm_cached_e2e}s ({cold_e2e/warm_cached_e2e:.1f}x) -> warm uncached {warm_uncached_e2e}s")

    finally:
        try: lf.close()
        except: pass
        if proc.poll() is None:
            print(f"killing {proc.pid}")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except:
                proc.kill()
                proc.wait(timeout=10)
        try: subprocess.run(["pkill","-f","python main.py.*--tpu"], timeout=5)
        except: pass
        time.sleep(5)

if __name__=="__main__":
    for prof in ["krea2", "krea2-1920x1080"]:
        print(f"\n========== Benchmark profile {prof} ==========")
        benchmark_profile(prof)
    print("\nAll done")

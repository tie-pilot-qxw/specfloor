"""One AR/DSpark pass on an exclusive GPU; owns and cleans up only its servers."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time

from serving_sweep_client import DATASETS, api, run_point

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/workspace/sglang-eval-env/bin/python"
OFFICIAL = "/workspace/tmp/ckpt_official_dspark"


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_state(gpu):
    def query(option, fields):
        return subprocess.check_output(["nvidia-smi", "-i", str(gpu), option + "=" + fields,
                                        "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
    state = query("--query-gpu", "uuid,memory.used,utilization.gpu")
    processes = query("--query-compute-apps", "pid,used_memory")
    return {"time_unix": time.time(), "gpu": state, "processes": processes,
            "pids": [int(line.split(",")[0]) for line in processes.splitlines() if line.strip()]}


def prepare(out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    rows = []
    for name, cap in DATASETS:
        source = ROOT / "eval_datasets" / (name + ".jsonl")
        texts = []
        for line in source.read_text().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            texts.append(d["turns"][0] if isinstance(d["turns"], list) else d["turns"])
        if len(texts) > cap:
            random.Random(980406).shuffle(texts)
            texts = texts[:cap]
        if len(texts) != cap:
            raise ValueError(f"Wrong count for {name}")
        for i, text in enumerate(texts):
            formatted = tok.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(formatted)
            rows.append({"dataset": name, "idx": i, "input_ids": ids, "text": formatted,
                         "prompt_sha1": hashlib.sha1(text.encode()).hexdigest(),
                         "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    with (out / "prepared_requests.jsonl").open("x") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def stop_server(pid):
    # PID is the start_new_session child this run recorded, never found by a port search.
    try:
        if os.getpgid(pid) != pid:
            raise RuntimeError("Recorded server is no longer its own process-group leader")
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(100):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--port", type=int, default=30486)
    ap.add_argument("--adopt-baseline", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepared = out / "prepared_requests.jsonl"
    rows = ([json.loads(s) for s in prepared.read_text().splitlines()]
            if prepared.exists() else prepare(out))
    if len(rows) != 3030:
        raise ValueError("Expected exactly 3030 requests")
    order = [32, 16, 8, 4, 1]
    sources = [Path(__file__), ROOT / "refine_exp/serving_sweep_client.py",
               ROOT / "refine_exp/serving_sweep_bootstrap/sitecustomize.py"]
    runtime = Path(PYTHON).parent.parent / "lib/python3.12/site-packages/sglang/srt"
    sources += [runtime / p for p in ["managers/scheduler.py", "observability/req_time_stats.py",
                "managers/scheduler_components/batch_result_processor.py",
                "models/dspark.py", "speculative/dspark_components/dspark_draft.py"]]
    source_hashes = {str(p): sha(p) for p in sources}
    snapshot = out / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for i, p in enumerate(sources):
        (snapshot / f"{i}_{p.name}").write_bytes(p.read_bytes())
    manifest = dict(arms=["baseline", "official"], concurrencies=[1, 4, 8, 16, 32],
        execution_order=order, temperature=0, max_new_tokens=2048, rows=3030,
        datasets=dict(DATASETS), seed=980406, gpu=args.gpu, mem_fraction_static=0.95,
        independent_passes=1, source_sha256=source_hashes,
        prepared_requests_sha256=sha(prepared),
        dataset_sha256={n: sha(ROOT / "eval_datasets" / (n + ".jsonl")) for n, _ in DATASETS},
        official_config_sha256=sha(Path(OFFICIAL) / "config.json"),
        timing="Per-task output tokens / exact perf_counter elapsed from request dispatch to all responses. "
               "Includes server tokenization, prefill and HTTP streaming. Preformatted chat; no prefix caching.",
        decode="Diagnostic per-request scheduler elapsed from prefill completion to finalization, "
               "including scheduling. N-1 tokens. Not GPU-only time or system throughput at C>1.",
        aggregation="Arithmetic mean of nine per-task speedups over AR at the same concurrency.",
        note="High concurrency first to validate the batch path; same order in both arms. "
             "Each point has two 32-request warmup waves capped at 64 tokens; all 3030 rows are measured.")
    write(out / "manifest.json", manifest)
    if args.prepare_only:
        print("Prepared", len(rows), "requests", flush=True)
        return
    base = f"http://127.0.0.1:{args.port}"
    cmd = json.loads((out / "baseline.command.json").read_text())
    env = dict(os.environ, HF_HOME="/workspace/.cache/huggingface", HF_HUB_OFFLINE="1",
               TMPDIR="/workspace/tmp", PYTHONPATH=str(ROOT / "refine_exp/serving_sweep_bootstrap"),
               SGLANG_RECORD_STEP_TIME="1", SGLANG_RAGGED_VERIFY_MODE="static",
               SGLANG_DFLASH_FUSE_CONV="1", SGLANG_DSPARK_FOLDED_LATTICE="1")
    all_results = {}
    pid = None
    monitor_stop = threading.Event()
    violation = []
    thread = None
    started = time.time()
    def status(**kw):
        write(out / "status.json", dict(started_unix=started, updated_unix=time.time(), **kw))
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for arm in ["baseline", "official"]:
            monitor_stop.clear()
            violation.clear()
            if arm == "baseline" and args.adopt_baseline:
                adopted = json.loads((out / "baseline.process.json").read_text())
                pid = adopted["pid"]
                if adopted["command"] != cmd or adopted["gpu"] != args.gpu:
                    raise RuntimeError("Adopted server configuration mismatch")
            else:
                while True:
                    state = gpu_state(args.gpu)
                    if not state["pids"] and int(state["gpu"].split(",")[1]) < 512:
                        break
                    status(state="waiting_for_gpu", arm=arm, gpu=state)
                    time.sleep(0.5)
                arm_cmd = cmd + (["--speculative-algorithm", "DSPARK",
                    "--speculative-draft-model-path", OFFICIAL, "--speculative-dspark-block-size", "7",
                    "--speculative-num-draft-tokens", "8"] if arm == "official" else [])
                write(out / f"{arm}.command.json", arm_cmd)
                with (out / f"{arm}.server.log").open("x") as log:
                    proc = subprocess.Popen(arm_cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, cwd=ROOT)
                pid = proc.pid
                write(out / f"{arm}.process.json", dict(pid=pid, pgid=pid, command=arm_cmd))
            status(state="starting_server", arm=arm, pid=pid)
            deadline = time.monotonic() + 600
            while True:
                os.kill(pid, 0)
                try:
                    api(base, "/health", timeout=2)
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{arm} startup")
                    time.sleep(1)
            state = gpu_state(args.gpu)
            if len(state["pids"]) != 1:
                raise RuntimeError(f"GPU not exclusive at measurement start: {state}")
            # Aggregate GPU memory can briefly include an exiting foreign process.
            # Require our sole compute process itself to own the near-full allocation.
            if int(state["processes"].split(",")[1]) < 74000:
                raise RuntimeError(f"Startup memory allocation short on 80GB GPU: {state}")
            expected = state["pids"]
            write(out / f"{arm}.gpu_ready.json", state)
            def monitor():
                with (out / f"{arm}.gpu_samples.jsonl").open("x", buffering=1) as log:
                    while not monitor_stop.is_set():
                        try:
                            observed = gpu_state(args.gpu)
                            log.write(json.dumps(observed) + "\n")
                            if observed["pids"] != expected:
                                violation.append(f"GPU process set changed: {observed}")
                                return
                        except Exception as exc:
                            violation.append(repr(exc))
                            return
                        monitor_stop.wait(2)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            def guard():
                if violation:
                    raise RuntimeError(violation[0])
            all_results[arm] = {}
            for c in order:
                for p in sources:
                    if sha(p) != source_hashes[str(p)]:
                        raise RuntimeError(f"Benchmark/runtime source changed mid-run: {p}")
                status(state="measuring", arm=arm, concurrency=c, pid=pid)
                print(f"START {arm} C={c}", flush=True)
                all_results[arm][str(c)] = run_point(base, rows, out, arm, c, guard)
                write(out / "results.json", all_results)
            guard()
            monitor_stop.set()
            thread.join(timeout=20)
            thread = None
            stop_server(pid)
            pid = None
        summary = {}
        for c in order:
            ar, spec = all_results["baseline"][str(c)], all_results["official"][str(c)]
            ratios = {name: spec[name]["system_output_tokens_per_second"] /
                      ar[name]["system_output_tokens_per_second"] for name, _ in DATASETS}
            summary[str(c)] = dict(per_task_speedup=ratios, macro_speedup=statistics.mean(ratios.values()),
                macro_accepted_length=statistics.mean(spec[n]["accepted_length"] for n, _ in DATASETS))
        write(out / "comparison.json", summary)
        status(state="complete", comparison=summary)
        print("COMPLETE", out, flush=True)
    except BaseException as exc:
        status(state="failed", error=repr(exc))
        raise
    finally:
        monitor_stop.set()
        if thread:
            thread.join(timeout=20)
        if pid is not None:
            stop_server(pid)


if __name__ == "__main__":
    main()

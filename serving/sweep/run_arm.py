"""Serve one arm on one GPU and measure it at each concurrency, in the archived format.

    python run_arm.py --prepared prepared_requests.jsonl --out runs/t0_ours \
        --arm ours --checkpoint <ours checkpoint> --temperature 0 --gpu 0 --port 30489

Arms: `baseline` (AR, no speculation), `official` (released DSpark) and `ours`.
By default the GPU must be empty when the server starts, the server must then own
at least 74 GB of it, and the GPU's process set is sampled every two seconds;
any change aborts the run. Timing is only meaningful under those conditions.
`--shared-gpu` drops these checks for functional smoke tests and marks the
manifest so the output cannot be mistaken for a timing run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import client
from common import (DATASETS, EXECUTION_ORDER, MAX_NEW_TOKENS, PREPARED_SHA256,
                    ROWS, SEED, TASK_COUNTS, gpu_state, read_jsonl, server_command, server_env,
                    sha, stop_server, write)

HERE = Path(__file__).resolve().parent
SOURCES = [HERE / "run_arm.py", HERE / "client.py", HERE / "common.py",
           HERE / "bootstrap" / "sitecustomize.py"]


def runtime_sources(python):
    """Runtime files whose hashes the archive records, resolved in the server's interpreter."""
    code = ("import json, sglang, pathlib; r = pathlib.Path(sglang.__file__).parent / 'srt'; "
            "print(json.dumps([str(r / p) for p in ['managers/scheduler.py', "
            "'observability/req_time_stats.py', 'managers/scheduler_components/batch_result_processor.py', "
            "'models/dspark.py', 'models/dflash.py', 'speculative/dspark_components/dspark_draft.py']]))")
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, required=True,
                    help="prepared_requests.jsonl(.gz) from prepare_requests.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", choices=["baseline", "official", "ours"], required=True)
    ap.add_argument("--checkpoint", type=Path, help="draft checkpoint; required unless --arm baseline")
    ap.add_argument("--temperature", type=int, choices=[0, 1], required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=EXECUTION_ORDER)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--python", type=Path, default=Path(sys.executable),
                    help="interpreter with the patched SGLang installed")
    ap.add_argument("--sglang-python-dir", type=Path, action="append", default=[],
                    help="prepend a patched checkout's python/ to the server's path instead of installing it")
    ap.add_argument("--mem-fraction", type=float, default=0.95)
    ap.add_argument("--shared-gpu", action="store_true",
                    help="skip GPU exclusivity checks (smoke tests only; not a timing run)")
    ap.add_argument("--tasks", nargs="+", choices=[n for n, _ in DATASETS],
                    help="measure only these tasks (smoke tests)")
    ap.add_argument("--limit", type=int, help="measure only the first N rows of each task (smoke tests)")
    args = ap.parse_args()
    if (args.arm == "baseline") != (args.checkpoint is None):
        ap.error("--checkpoint is required for official/ours and not allowed for baseline")
    subset = bool(args.tasks or args.limit)
    if subset and not args.shared_gpu:
        ap.error("--tasks/--limit are for smoke tests; combine them with --shared-gpu")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    prepared, raw = read_jsonl(args.prepared)
    prepared_sha = hashlib.sha256(raw).hexdigest()
    if len(prepared) != ROWS or prepared_sha != PREPARED_SHA256:
        raise RuntimeError(f"prepared requests differ from the archived set ({prepared_sha})")
    rows, counts = prepared, dict(TASK_COUNTS)
    if subset:
        keep = set(args.tasks or TASK_COUNTS)
        counts = {n: min(c, args.limit or c) for n, c in TASK_COUNTS.items() if n in keep}
        rows = [r for r in prepared if r["dataset"] in counts and r["idx"] < counts[r["dataset"]]]

    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    cmd = server_command(args.python, args.gpu, args.port, checkpoint, args.mem_fraction)
    env = server_env(args.sglang_python_dir)
    runtime = runtime_sources(args.python) if not args.sglang_python_dir else [
        str(d / "sglang/srt" / p) for d in args.sglang_python_dir[:1]
        for p in ["managers/scheduler.py", "observability/req_time_stats.py",
                  "managers/scheduler_components/batch_result_processor.py",
                  "models/dspark.py", "models/dflash.py", "speculative/dspark_components/dspark_draft.py"]]
    source_hashes = {str(p): sha(p) for p in [*SOURCES, *map(Path, runtime)]}
    manifest = dict(
        arms=[args.arm], concurrencies=sorted(args.concurrency), execution_order=args.concurrency,
        temperature=args.temperature, max_new_tokens=MAX_NEW_TOKENS, rows=len(rows),
        datasets=counts, seed=SEED, gpu=args.gpu, mem_fraction_static=args.mem_fraction,
        independent_passes=1, source_sha256=source_hashes, prepared_requests_sha256=prepared_sha,
        checkpoint=str(checkpoint) if checkpoint else None,
        checkpoint_config_sha256=sha(checkpoint / "config.json") if checkpoint else None,
        shared_gpu=args.shared_gpu, subset=subset,
        timing="Per-task output tokens / exact perf_counter elapsed from request dispatch to all responses. "
               "Includes server tokenization, prefill and HTTP streaming. Preformatted chat; no prefix caching.",
        decode="Diagnostic per-request scheduler elapsed from prefill completion to finalization, "
               "including scheduling. N-1 tokens. Not GPU-only time or system throughput at C>1.",
        aggregation="Arithmetic mean of nine per-task speedups over AR at the same concurrency.",
        note="Each point has two 32-request warmup waves capped at 64 tokens; every listed row is measured.")
    if args.temperature == 1:
        manifest["sampling_seed"] = SEED
    write(out / "manifest.json", manifest)
    write(out / f"{args.arm}.command.json", cmd)

    base = f"http://127.0.0.1:{args.port}"
    proc = thread = None
    monitor_stop = threading.Event()
    violation = []
    started = time.time()

    def status(**kw):
        write(out / "status.json", dict(arm=args.arm, gpu=args.gpu, started_unix=started,
                                       updated_unix=time.time(), **kw))

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while not args.shared_gpu:
            state = gpu_state(args.gpu)
            if not state["pids"] and int(state["gpu"].split(",")[1]) < 512:
                break
            status(state="waiting_for_gpu", gpu_state=state)
            time.sleep(.5)
        with (out / f"{args.arm}.server.log").open("x") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        write(out / f"{args.arm}.process.json", dict(pid=proc.pid, pgid=proc.pid, command=cmd))
        status(state="starting_server", pid=proc.pid)
        deadline = time.monotonic() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {proc.returncode}")
            try:
                client.api(base, "/health", timeout=2)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                time.sleep(1)
        state = gpu_state(args.gpu)
        write(out / f"{args.arm}.gpu_ready.json", state)
        if not args.shared_gpu:
            # Aggregate memory can briefly include an exiting foreign process, so the
            # server itself must own the near-full allocation.
            if len(state["pids"]) != 1 or int(state["processes"].split(",")[1]) < 74000:
                raise RuntimeError(f"GPU must be exclusive and allocated near-full: {state}")

            def monitor():
                with (out / f"{args.arm}.gpu_samples.jsonl").open("x", buffering=1) as fh:
                    while not monitor_stop.is_set():
                        try:
                            observed = gpu_state(args.gpu)
                            fh.write(json.dumps(observed) + "\n")
                            if observed["pids"] != state["pids"]:
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
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited: {proc.returncode}")

        results = {}
        for c in args.concurrency:
            for path, expected in source_hashes.items():
                if sha(path) != expected:
                    raise RuntimeError(f"Source changed during run: {path}")
            status(state="measuring", concurrency=c, pid=proc.pid)
            print(f"START {args.arm} T={args.temperature} GPU={args.gpu} C={c}", flush=True)
            results[str(c)] = client.run_point(base, rows, out, args.arm, c, args.temperature,
                                               guard, prepared=prepared, counts=counts)
            write(out / "results.json", results)
        guard()
        status(state="complete")
        print("COMPLETE", args.arm, out, flush=True)
    except BaseException as exc:
        status(state="failed", error=repr(exc))
        raise
    finally:
        monitor_stop.set()
        if thread:
            thread.join(timeout=20)
        if proc:
            stop_server(proc.pid)


if __name__ == "__main__":
    main()

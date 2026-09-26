"""Run the remaining temperature-1 C=32/16/8/4 points; C=1 is supplied by the existing queue."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from run_serving_sweep import ROOT, PYTHON, OFFICIAL, sha, stop_server, write
from serving_temp1_client import run_point
from resume_serving_sweep import gpu_state

OURS = "/workspace/checkpoints/deepspec/attnconv_b7_qwen3_4b_10ep/step_26160"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--arm", choices=["baseline", "official", "ours"], required=True)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve()
    rows = [json.loads(s) for s in (source / "prepared_requests.jsonl").read_text().splitlines()]
    if len(rows) != 3030:
        raise ValueError("Expected 3030 requests")
    checkpoint = None if args.arm == "baseline" else (OFFICIAL if args.arm == "official" else OURS)
    manifest = json.loads((source / "manifest.json").read_text())
    if sha(source / "prepared_requests.jsonl") != manifest["prepared_requests_sha256"]:
        raise RuntimeError("Prepared requests changed from AR manifest")
    # Match the actual AR implementation, not just similarly named flags.
    for path, expected in manifest["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Source differs from AR: {path}")
    manifest.update(arms=[args.arm], gpu=args.gpu, reference_ar=str(source),
                    checkpoint=checkpoint, checkpoint_config_sha256=sha(Path(checkpoint) / "config.json") if checkpoint else None,
                    cross_gpu_authorized=True, parallel_runner_sha256=sha(__file__))
    manifest.update(temperature=1.0, sampling_seed=980406, concurrencies=[4, 8, 16, 32], execution_order=[32, 16, 8, 4],
                    note="One temperature-1 pass at C=32/16/8/4; existing C=1 is not rerun; sampling_seed=980406.",
                    telemetry_timeout_s=60)
    for p in [Path(__file__), ROOT / "refine_exp/serving_temp1_client.py", ROOT / "refine_exp/resume_serving_sweep.py"]:
        manifest["source_sha256"][str(p)] = sha(p)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for i, path in enumerate(manifest["source_sha256"]):
        (snapshot / f"{i}_{Path(path).name}").write_bytes(Path(path).read_bytes())
    write(out / "manifest.json", manifest)
    cmd = json.loads((source / "baseline.command.json").read_text())
    cmd[cmd.index("--base-gpu-id") + 1] = str(args.gpu)
    cmd[cmd.index("--port") + 1] = str(args.port)
    if checkpoint:
        cmd += ["--speculative-algorithm", "DSPARK", "--speculative-draft-model-path", checkpoint,
            "--speculative-dspark-block-size", "7", "--speculative-num-draft-tokens", "8"]
    write(out / f"{args.arm}.command.json", cmd)
    env = dict(os.environ, HF_HOME="/workspace/.cache/huggingface", HF_HUB_OFFLINE="1",
               TMPDIR="/workspace/tmp", PYTHONPATH=str(ROOT / "refine_exp/serving_sweep_bootstrap"),
               SGLANG_RECORD_STEP_TIME="1", SGLANG_RAGGED_VERIFY_MODE="static",
               SGLANG_DFLASH_FUSE_CONV="1", SGLANG_DSPARK_FOLDED_LATTICE="1")
    base = f"http://127.0.0.1:{args.port}"
    from serving_sweep_client import api
    proc = None
    monitor_stop = threading.Event()
    violation = []
    thread = None
    started = time.time()
    def status(**kw):
        write(out / "status.json", dict(arm=args.arm, gpu=args.gpu, started_unix=started,
                                       updated_unix=time.time(), **kw))
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while True:
            state = gpu_state(args.gpu)
            if not state["pids"] and int(state["gpu"].split(",")[1]) < 512:
                break
            status(state="waiting_for_gpu", gpu_state=state)
            time.sleep(.5)
        with (out / f"{args.arm}.server.log").open("x") as log:
            proc = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        write(out / f"{args.arm}.process.json", dict(pid=proc.pid, pgid=proc.pid, command=cmd))
        status(state="starting_server", pid=proc.pid)
        deadline = time.monotonic() + 600
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {proc.returncode}")
            try:
                api(base, "/health", timeout=2)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                time.sleep(1)
        state = gpu_state(args.gpu)
        if len(state["pids"]) != 1 or int(state["processes"].split(",")[1]) < 74000:
            raise RuntimeError(f"GPU must be exclusive and allocated near-full: {state}")
        expected_pids = state["pids"]
        write(out / f"{args.arm}.gpu_ready.json", state)
        def monitor():
            with (out / f"{args.arm}.gpu_samples.jsonl").open("x", buffering=1) as fh:
                while not monitor_stop.is_set():
                    try:
                        observed = gpu_state(args.gpu)
                        fh.write(json.dumps(observed) + "\n")
                        if observed["pids"] != expected_pids:
                            violation.append(f"GPU process set changed: {observed}")
                            return
                    except Exception as exc:
                        violation.append(repr(exc))
                        return
                    monitor_stop.wait(2)
        def guard():
            if violation:
                raise RuntimeError(violation[0])
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        results = {}
        for c in manifest["execution_order"]:
            for path, expected in manifest["source_sha256"].items():
                if sha(path) != expected:
                    raise RuntimeError(f"Source changed during run: {path}")
            status(state="measuring", concurrency=c, pid=proc.pid)
            print(f"START {args.arm} GPU={args.gpu} C={c}", flush=True)
            results[str(c)] = run_point(base, rows, out, args.arm, c, guard)
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

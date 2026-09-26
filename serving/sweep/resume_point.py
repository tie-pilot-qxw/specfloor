"""Finish an interrupted point at whole-task boundaries, keeping the original task timers.

    python resume_point.py --run runs/t0_ours --arm ours --concurrency 1

Tasks already saved in `<arm>.c<C>.jsonl` are validated (whole, identity-matched,
and reproducing their saved summary) and kept. The server is restarted with the
recorded command and only the remaining tasks are measured, each under a fresh
timer inside one server session. Every recovery writes its own directory with the
original files, a provenance manifest and GPU samples, so the interruption stays
visible in the record. This is how the archived temperature-0 and temperature-1
C=1 points that were interrupted were completed.
"""
import argparse
import hashlib
import fcntl
import json
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time

import client
from common import gpu_state, read_jsonl, server_env, sha, stop_server, write


def validate_saved(rows, summaries, prepared, arm, counts):
    """A saved task must be whole, identity-matched, and reproduce its summary."""
    if set(summaries) - set(counts):
        raise ValueError("Unknown saved task")
    if {r["dataset"] for r in rows} != set(summaries):
        raise ValueError("Saved rows and summaries disagree")
    identity = lambda r: (r["idx"], r["prompt_sha1"], r["input_ids_sha256"])
    for name, summary in summaries.items():
        selected = [r for r in rows if r["dataset"] == name]
        expected = [r for r in prepared if r["dataset"] == name][:counts[name]]
        if len(selected) != counts[name] or sorted(map(identity, selected)) != sorted(map(identity, expected)):
            raise ValueError(f"Incomplete or changed task: {name}")
        if client.summarize(selected, summary["elapsed_s"], arm) != summary:
            raise ValueError(f"Summary differs from raw rows: {name}")
    return [(n, c) for n, c in counts.items() if n not in summaries]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="the run_arm.py output directory")
    ap.add_argument("--arm", choices=["baseline", "official", "ours"], required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--prepared", type=Path, required=True)
    ap.add_argument("--sglang-python-dir", type=Path, action="append", default=[])
    args = ap.parse_args()
    out = args.run.resolve()
    lock = (out / "recovery.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    arm, c = args.arm, args.concurrency
    prefix = f"{arm}.c{c}"
    if (out / f"{prefix}.complete.json").exists():
        raise RuntimeError(f"C={c} already complete; refusing to rerun")
    manifest = json.loads((out / "manifest.json").read_text())
    temperature = 0 if manifest["temperature"] == 0 else 1
    counts = manifest["datasets"]
    prepared, raw = read_jsonl(args.prepared)
    if hashlib.sha256(raw).hexdigest() != manifest["prepared_requests_sha256"]:
        raise RuntimeError("Prepared requests changed")
    saved = [json.loads(s) for s in (out / f"{prefix}.jsonl").read_text().splitlines()]
    tags = client.row_tags(temperature)
    if any(any(r.get(k) != v for k, v in tags.items()) for r in saved):
        raise ValueError("Saved rows do not match the run's temperature protocol")
    summaries = json.loads((out / f"{prefix}.summary.json").read_text())
    remaining = validate_saved(saved, summaries, prepared, arm, counts)
    gpu = manifest["gpu"]
    recovery = out / ("recovery_" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()))
    recovery.mkdir()
    for name in ["status.json", f"{prefix}.jsonl", f"{prefix}.summary.json", "results.json"]:
        if (out / name).exists():
            shutil.copy2(out / name, recovery / ("original." + name))
    provenance = dict(started_unix=time.time(), retained_tasks=list(summaries),
        remaining_tasks=[n for n, _ in remaining], temperature=temperature,
        runner_sha256=sha(__file__), original_manifest=str(out / "manifest.json"),
        note=f"C={c} resumes at whole-task boundaries; no elapsed timers span server restarts.")
    write(recovery / "manifest.json", provenance)
    write(out / f"{prefix}.recovery.json", dict(provenance, path=str(recovery)))

    def verify_sources():
        for path, expected in manifest["source_sha256"].items():
            if Path(path).exists() and sha(path) != expected:
                raise RuntimeError(f"Source changed: {path}")
    verify_sources()
    cmd = json.loads((out / f"{arm}.command.json").read_text())
    base = f"http://127.0.0.1:{cmd[cmd.index('--port') + 1]}"
    env = server_env(args.sglang_python_dir)
    exclusive = not manifest.get("shared_gpu")
    proc = thread = None
    stop = threading.Event()
    violations = []

    def status(**kw):
        payload = dict(arm=arm, gpu=gpu, concurrency=c, updated_unix=time.time(),
                       recovery_path=str(recovery), **kw)
        write(out / "status.json", payload)
        write(recovery / "status.json", payload)

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while exclusive:
            state = gpu_state(gpu)
            if not state["pids"] and int(state["gpu"].split(",")[1]) < 512:
                break
            status(state="waiting_for_gpu", observed=state)
            time.sleep(2)
        with (out / f"{arm}.server.log").open("a") as log:
            log.write(f"\n{arm} C={c} recovery: {recovery}\n")
            log.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        write(recovery / "server.process.json", dict(pid=proc.pid, command=cmd))
        status(state="starting_server", pid=proc.pid)
        deadline = time.monotonic() + 900
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited: {proc.returncode}")
            try:
                client.api(base, "/health", timeout=2)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise TimeoutError("Server startup")
                time.sleep(1)
        ready = gpu_state(gpu)
        write(recovery / "gpu_ready.json", ready)
        if exclusive:
            if len(ready["pids"]) != 1 or int(ready["processes"].split(",")[1]) < 74000:
                raise RuntimeError(f"GPU not exclusive/near-full: {ready}")

            def monitor():
                with (recovery / "gpu_samples.jsonl").open("x", buffering=1) as log:
                    while not stop.is_set():
                        try:
                            observed = gpu_state(gpu)
                            log.write(json.dumps(observed) + "\n")
                            if observed["pids"] != ready["pids"]:
                                raise RuntimeError(f"GPU process set changed: {observed}")
                        except Exception as exc:
                            violations.append(repr(exc))
                            return
                        stop.wait(2)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()

        def guard():
            if violations:
                raise RuntimeError(violations[0])
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited: {proc.returncode}")

        write(recovery / "before.json", client.api(base, "/get_server_info"))
        client.api(base, "/flush_cache")
        client.warm(base, prepared, arm, c, temperature)
        client.api(base, "/flush_cache")
        write(recovery / "measured_before.json", client.api(base, "/get_server_info"))
        for name, count in remaining:
            verify_sources()
            guard()
            status(state="measuring", pid=proc.pid, dataset=name)
            selected = [r for r in prepared if r["dataset"] == name][:count]
            measured, elapsed = client.measure_task(base, selected, c, temperature)
            guard()
            summary = client.summarize(measured, elapsed, arm)
            for row in measured:
                row.update(arm=arm, concurrency=c, **tags)
            # Store each recovered task on its own before touching the canonical files.
            with (recovery / f"{name}.jsonl").open("x") as fh:
                fh.writelines(json.dumps(row) + "\n" for row in measured)
            write(recovery / f"{name}.summary.json", summary)
            with (out / f"{prefix}.jsonl").open("a") as fh:
                fh.writelines(json.dumps(row) + "\n" for row in measured)
            summaries[name] = summary
            saved.extend(measured)
            write(out / f"{prefix}.summary.json", summaries)
            print(f"RECOVERED C={c} {name}: {count} rows {elapsed:.6f}s", flush=True)
        if validate_saved(saved, summaries, prepared, arm, counts):
            raise RuntimeError("Missing tasks after recovery")
        guard()
        after = client.api(base, "/get_server_info")
        write(recovery / "after.json", after)
        write(out / f"{prefix}.after.json", dict(after, recovery_note=str(recovery)))
        results_path = out / "results.json"
        results = json.loads(results_path.read_text()) if results_path.exists() else {}
        results[str(c)] = summaries
        write(results_path, results)
        write(out / f"{prefix}.complete.json", dict(complete=True, rows=len(saved),
              completed_unix=time.time(), concurrency=c, arm=arm, recovery_path=str(recovery)))
        status(state="complete")
    except BaseException as exc:
        status(state="failed", error=repr(exc))
        raise
    finally:
        stop.set()
        if proc:
            stop_server(proc.pid)
        if thread:
            thread.join(timeout=5)


if __name__ == "__main__":
    main()

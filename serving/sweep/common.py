"""Shared pieces of the serving sweep: tasks, the server command, its environment.

The server flags and environment below are the ones every archived sweep point
was launched with (see results/serving_sweep_20260922/runs/*/*.command.json).
Only the interpreter, GPU, port and checkpoint paths are parameters.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

DATASETS = [("gsm8k", 500), ("math500", 500), ("aime25", 30),
            ("humaneval", 164), ("mbpp", 256), ("livecodebench", 500),
            ("mt-bench", 80), ("alpaca", 500), ("arena-hard-v2", 500)]
TASK_COUNTS = dict(DATASETS)
ROWS = sum(TASK_COUNTS.values())            # 3030
SEED = 980406                               # prompt shuffle and temperature-1 sampling seed
MAX_NEW_TOKENS = 2048
CONCURRENCIES = [1, 4, 8, 16, 32]
EXECUTION_ORDER = [32, 16, 8, 4, 1]         # high concurrency first validates the batch path
TARGET = "Qwen/Qwen3-4B"
PREPARED_SHA256 = "b5efd513e563d20f5880fec8cf25c3e2afb1899bf19ae1a96d0334811bb24831"

BOOTSTRAP = Path(__file__).resolve().parent / "bootstrap"

# SGLANG_DFLASH_FUSE_CONV defaults to 0 in the patched runtime and every archived
# point set it to 1. The other two DSpark switches are on by default; setting them
# explicitly keeps a launch independent of those defaults.
SERVER_ENV = {
    "SGLANG_RECORD_STEP_TIME": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DFLASH_FUSE_CONV": "1",
    "SGLANG_DSPARK_FOLDED_LATTICE": "1",
}


def server_command(python, gpu, port, checkpoint=None, mem_fraction=0.95):
    """The archived launch command; `checkpoint=None` is the AR baseline."""
    cmd = [str(python), "-m", "sglang.launch_server", "--model-path", TARGET,
           "--mem-fraction-static", str(mem_fraction), "--max-running-requests", "32",
           "--cuda-graph-max-bs-decode", "32", "--cuda-graph-backend-prefill", "disabled",
           "--chunked-prefill-size", "8192", "--max-prefill-tokens", "16384",
           "--disable-radix-cache", "--enable-metrics", "--dtype", "bfloat16",
           "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port),
           "--base-gpu-id", str(gpu)]
    if checkpoint is not None:
        cmd += ["--speculative-algorithm", "DSPARK", "--speculative-draft-model-path",
                str(checkpoint), "--speculative-dspark-block-size", "7",
                "--speculative-num-draft-tokens", "8"]
    return cmd


def server_env(extra_pythonpath=()):
    """Server environment: the timing bootstrap first, then any patched checkout."""
    path = [str(BOOTSTRAP), *map(str, extra_pythonpath)]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(path), **SERVER_ENV)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def gpu_state(gpu, timeout=60):
    def query(option, fields):
        return subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), option + "=" + fields,
             "--format=csv,noheader,nounits"], text=True, timeout=timeout).strip()
    started = time.time()
    state = query("--query-gpu", "uuid,memory.used,utilization.gpu")
    processes = query("--query-compute-apps", "pid,used_memory")
    return {"time_unix": time.time(), "query_started_unix": started, "gpu": state,
            "processes": processes,
            "pids": [int(line.split(",")[0]) for line in processes.splitlines() if line.strip()]}


def stop_server(pid):
    """Stop the process group this run started; never a process found by port."""
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


def read_jsonl(path):
    import gzip
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()], raw

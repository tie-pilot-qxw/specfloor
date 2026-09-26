"""Short real-serving CUDA-event probe. Only kills the server group it starts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/workspace/sglang-eval-env/bin/python"
CHECKPOINTS = {
    "official": "/workspace/tmp/ckpt_official_dspark",
    "ours": "/workspace/checkpoints/deepspec/attnconv_b7_qwen3_4b_10ep/step_26160",
}


def get(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, default=30483)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["official", "ours"], choices=list(CHECKPOINTS))
    ap.add_argument("--allow-idle-resident", action="store_true",
                    help="Allow resident allocations only when sampled GPU utilization is idle")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--contexts", type=int, nargs="+", default=[512, 1024])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    if len(set(args.batches)) != len(args.batches) or any(b < 1 or b > 32 for b in args.batches):
        raise ValueError("batches must be distinct values in 1..32")
    if any(c <= 0 or c * max(args.batches) > 32768 for c in args.contexts):
        raise ValueError("context must fit one prefill batch of at most 32768 tokens")
    manifest = dict(gpu=args.gpu, batches=args.batches, contexts=args.contexts,
                    arms=args.arms, allow_idle_resident=args.allow_idle_resident,
                    temperature=0, gamma=7, warmup=12, iterations=64,
                    checkpoint_paths=CHECKPOINTS, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    method="fixed-state replay of actual SGLang draft proposal; CUDA-event intervals")
    source_paths = [Path(__file__), ROOT / "refine_exp/draft_forward_probe_server.py",
                    ROOT / "refine_exp/draft_probe_bootstrap/sitecustomize.py"]
    manifest["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest["checkpoint_config_sha256"] = {a: hashlib.sha256((Path(p)/"config.json").read_bytes()).hexdigest()
                                            for a,p in CHECKPOINTS.items()}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", local_files_only=True)
    ids = tok.encode("Explain how a computer processes information and performs arithmetic. ",
                     add_special_tokens=False)
    base = f"http://127.0.0.1:{args.port}"
    for arm in args.arms:
        gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        used, util = map(int, gpu.strip().split(","))
        if (used > 512 and not args.allow_idle_resident) or util > 5 or used > 51000:
            raise RuntimeError(f"GPU {args.gpu} is not free: {gpu.strip()}")
        (args.out / f"{arm}.gpu_before.txt").write_text(gpu)
        env = dict(os.environ, HF_HOME="/workspace/.cache/huggingface", HF_HUB_OFFLINE="1",
                   TMPDIR="/workspace/tmp", DRAFT_PROBE_ARM=arm,
                   DRAFT_PROBE_OUT=str(args.out / f"{arm}.jsonl"),
                   DRAFT_PROBE_BATCHES=",".join(map(str,args.batches)),
                   DRAFT_PROBE_CONTEXTS=",".join(map(str,args.contexts)),
                   SGLANG_RAGGED_VERIFY_MODE="static", SGLANG_DFLASH_FUSE_CONV="1",
                   SGLANG_DSPARK_FOLDED_LATTICE="1")
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "refine_exp/draft_probe_bootstrap"),
                                           str(ROOT / "refine_exp")])
        cmd = [PYTHON, "-m", "sglang.launch_server",
               "--model-path", "Qwen/Qwen3-4B", "--speculative-algorithm", "DSPARK",
               "--speculative-draft-model-path", CHECKPOINTS[arm],
               "--speculative-dspark-block-size", "7", "--speculative-num-draft-tokens", "8",
               "--mem-fraction-static", "0.92" if args.allow_idle_resident else "0.8",
               "--max-total-tokens", "60000" if args.allow_idle_resident else "100000",
               "--max-running-requests", "32", "--cuda-graph-max-bs-decode", "32",
               "--cuda-graph-backend-prefill", "disabled", "--chunked-prefill-size", "32768",
               "--max-prefill-tokens", "32768",
               "--disable-radix-cache", "--dtype", "bfloat16", "--trust-remote-code",
               "--host", "127.0.0.1", "--port", str(args.port), "--base-gpu-id", str(args.gpu)]
        (args.out / f"{arm}.command.json").write_text(json.dumps(cmd, indent=2))
        with (args.out / f"{arm}.server.log").open("w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, cwd=ROOT)
            try:
                deadline = time.monotonic() + 600
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{arm} server exited: inspect {args.out / (arm+'.server.log')}")
                    try:
                        get(base + "/health", timeout=2)
                        break
                    except Exception:
                        if time.monotonic() > deadline:
                            raise TimeoutError("server startup")
                        time.sleep(2)
                print(f"{arm}: server ready", flush=True)
                for context in args.contexts:
                    prompt = (ids * (context // len(ids) + 1))[:context]
                    for batch in args.batches:
                        body = {"input_ids": [prompt for _ in range(batch)], "stream": False,
                                "sampling_params": {"temperature": 0, "max_new_tokens": 32,
                                                    "ignore_eos": True}}
                        result = get(base + "/generate", body, timeout=300)
                        (args.out / f"{arm}.b{batch}.ctx{context}.response.json").write_text(
                            json.dumps(result))
                        print(f"{arm}: completed B={batch}, prefix={context}", flush=True)
                info = get(base + "/get_server_info", timeout=30)
                (args.out / f"{arm}.server_info.json").write_text(json.dumps(info, indent=2))
                samples = args.out / f"{arm}.jsonl"
                if not samples.exists():
                    raise RuntimeError("No CUDA-event probe records were collected")
                seen = {(r["batch_size"], r["context_label"]) for r in
                        map(json.loads, samples.read_text().splitlines())}
                if seen != {(b,c) for b in args.batches for c in args.contexts}:
                    raise RuntimeError(f"Missing actual fixed-batch probe points: {seen}")
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                # Scheduler CUDA allocations can outlive the HTTP parent's exit by
                # a few seconds. Wait for our added allocation to disappear.
                for _ in range(30):
                    now = int(subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip())
                    if now <= used + 512:
                        break
                    time.sleep(1)
    print(f"DONE: {args.out}", flush=True)


if __name__ == "__main__":
    main()

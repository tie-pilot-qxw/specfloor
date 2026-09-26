"""Exact wall-clock serving throughput, plus separately labelled decode diagnostics.

One module for both temperatures. At temperature 0 a request carries only
temperature and max_new_tokens; at temperature 1 it also carries the fixed
sampling seed. These are the request bodies of the archived temperature-0 and
temperature-1 clients (results/serving_sweep_20260922/sources/).
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

from common import DATASETS, MAX_NEW_TOKENS, SEED, TASK_COUNTS


def api(base, path, body=None, timeout=60):
    req = urllib.request.Request(base + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {"text": data.decode()}


def sampling_params(temperature, max_tokens):
    if temperature == 0:
        return {"temperature": 0, "max_new_tokens": max_tokens}
    if temperature == 1:
        return {"temperature": 1.0, "max_new_tokens": max_tokens, "sampling_seed": SEED}
    raise ValueError("The protocol has temperature 0 and temperature 1 only")


def row_tags(temperature):
    return {"temperature": 0} if temperature == 0 else {"temperature": 1.0, "sampling_seed": SEED}


def request(base, row, temperature, max_tokens=MAX_NEW_TOKENS):
    body = {"text": row["text"], "stream": True,
            "sampling_params": sampling_params(temperature, max_tokens)}
    req = urllib.request.Request(base + "/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first_time = first_count = None
    last = None
    with urllib.request.urlopen(req, timeout=3600) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            chunk = raw[5:].strip()
            if chunk == b"[DONE]":
                break
            last = json.loads(chunk)
            if first_time is None and last.get("meta_info", {}).get("completion_tokens", 0) > 0:
                first_time = time.perf_counter() - started
                first_count = last["meta_info"]["completion_tokens"]
    elapsed = time.perf_counter() - started
    if last is None or "meta_info" not in last:
        raise RuntimeError(f"Missing response: {row['dataset']}:{row['idx']}")
    meta = last["meta_info"]
    if not meta.get("finish_reason") or meta["finish_reason"].get("type") not in ("stop", "length"):
        raise RuntimeError(f"Bad finish: {meta}")
    return {"dataset": row["dataset"], "idx": row["idx"],
            "prompt_sha1": row["prompt_sha1"], "input_ids_sha256": row["input_ids_sha256"],
            "expected_prompt_tokens": len(row["input_ids"]),
            "wall_s": elapsed, "ttft_s": first_time,
            "first_chunk_completion_tokens": first_count,
            "sha1": hashlib.sha1(last.get("text", "").encode()).hexdigest(),
            "meta_info": meta, "completion_tokens": meta["completion_tokens"],
            "spec_verify_ct": meta.get("spec_verify_ct", 0),
            "scheduler_decode_elapsed_s": meta.get("sweep_scheduler_decode_elapsed_s")}


def summarize(rows, elapsed, arm):
    if elapsed <= 0 or not rows:
        raise ValueError("Empty or non-positive timing")
    keys = {(r["dataset"], r["idx"], r["prompt_sha1"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Duplicate requests")
    for r in rows:
        m = r["meta_info"]
        if m["prompt_tokens"] != r["expected_prompt_tokens"]:
            raise ValueError("Input token count changed")
        if m.get("num_retractions", 0) != 0:
            raise ValueError("Request retracted; memory capacity affected timing")
        decode_s = r["scheduler_decode_elapsed_s"]
        if (r["completion_tokens"] < 1 or decode_s is None or decode_s < 0
                or (r["completion_tokens"] > 1 and decode_s == 0)):
            raise ValueError("Missing usable scheduler decode timing")
        if m.get("cached_tokens", 0) != 0:
            raise ValueError("Unexpected prefix cache reuse")
        if arm != "baseline" and r["completion_tokens"] > 1 and not (r["spec_verify_ct"] or 0) > 0:
            raise ValueError("Speculation not active")
    n = sum(r["completion_tokens"] for r in rows)
    v = sum(r["spec_verify_ct"] or 0 for r in rows)
    decode = sum(r["scheduler_decode_elapsed_s"] for r in rows)
    return {"rows": len(rows), "completion_tokens": n, "elapsed_s": elapsed,
            "system_output_tokens_per_second": n / elapsed,
            "verification_steps": v, "accepted_length": n / v if v else None,
            "summed_scheduler_decode_elapsed_s": decode,
            "post_prefill_output_tokens": n - len(rows),
            "per_request_decode_tokens_per_second": (n - len(rows)) / decode,
            "decode_note": "Scheduler prefill-finished to completion, including scheduling and finalization. "
                           "Subtract one prefill output token per request. At C>1 this is NOT system throughput."}


def warmup_rows(prepared):
    """Four rows per task, interleaved, first 32: the archived warmup set."""
    by_task = {name: [r for r in prepared if r["dataset"] == name] for name, _ in DATASETS}
    return [by_task[name][i] for i in range(4) for name, _ in DATASETS][:32]


def warm(base, prepared, arm, concurrency, temperature):
    """Two warmup waves capped at 64 tokens; their rows are validated, not recorded."""
    rows = warmup_rows(prepared)
    for _ in range(2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            done = list(pool.map(lambda row: request(base, row, temperature, 64), rows))
        summarize(done, sum(r["wall_s"] for r in done), arm)


def measure_task(base, selected, concurrency, temperature):
    # Chat formatting is done; server tokenization remains inside the timed request.
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        measured = list(pool.map(lambda row: request(base, row, temperature), selected))
    return measured, time.perf_counter() - start


def run_point(base, rows, out, arm, concurrency, temperature, guard=lambda: None,
              prepared=None, counts=TASK_COUNTS):
    """Measure every task of `rows` at one concurrency; `prepared` supplies warmup rows."""
    out = Path(out)
    prefix = f"{arm}.c{concurrency}"
    (out / f"{prefix}.before.json").write_text(json.dumps(api(base, "/get_server_info")))
    # Prefix caching is disabled; flush still verifies that no old requests remain.
    api(base, "/flush_cache")
    warm(base, prepared or rows, arm, concurrency, temperature)
    api(base, "/flush_cache")
    guard()
    (out / f"{prefix}.measured_before.json").write_text(json.dumps(api(base, "/get_server_info")))
    results = {}
    with (out / f"{prefix}.jsonl").open("x") as fh:
        for name, count in counts.items():
            selected = [r for r in rows if r["dataset"] == name]
            if len(selected) != count:
                raise ValueError(f"Incomplete {name}: {len(selected)} != {count}")
            guard()
            measured, elapsed = measure_task(base, selected, concurrency, temperature)
            guard()
            for r in measured:
                r.update(arm=arm, concurrency=concurrency, **row_tags(temperature))
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            results[name] = summarize(measured, elapsed, arm)
            (out / f"{prefix}.summary.json").write_text(json.dumps(results, indent=2))
            print(f"{arm} T={temperature} C={concurrency} {name}: {len(measured)} rows, "
                  f"{elapsed:.6f}s, {results[name]['system_output_tokens_per_second']:.3f} tok/s",
                  flush=True)
    (out / f"{prefix}.after.json").write_text(json.dumps(api(base, "/get_server_info")))
    guard()
    (out / f"{prefix}.complete.json").write_text(json.dumps({"complete": True, "rows": len(rows),
        "completed_unix": time.time(), "concurrency": concurrency, "arm": arm}))
    return results

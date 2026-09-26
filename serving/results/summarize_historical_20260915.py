"""Summarize the September 15 single-request serving runs (historical; not in the paper).

    python summarize_historical_20260915.py      # reads historical_20260915/, writes generated/

These sequential temperature-0/1 timings and the three-launch temperature-1
acceptance were the paper's serving numbers until the September 22 sweep
(serving_sweep_20260922/) replaced them. They ran on the runtime revision
8f51a8a (see serving/sglang/README.md). The records are kept so the superseded
values can still be checked; nothing in the paper reads them now.
Row identities include dataset, index, and prompt hash, so duplicate prompts
remain in the estimates.
"""

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from summarize_accept_evals import summarize


TASK_COUNTS = {
    "gsm8k": 500, "math500": 500, "aime25": 30, "humaneval": 164,
    "mbpp": 256, "livecodebench": 500, "mt-bench": 80, "alpaca": 500,
    "arena-hard-v2": 500,
}


# Canonical timing comparisons: same-session T0 control and historical T1 control.
SERVING_SOURCES = {
    0: {
        "official": "t0/official.jsonl.gz",
        "optimized": "t0/ours.jsonl.gz",
    },
    1: {
        "official": "t1/official.jsonl.gz",
        "optimized": "t1/ours.jsonl.gz",
    },
}

T1_ACCEPTANCE_SOURCES = {
    arm: [f"t1_acceptance/{arm}_r{rep}.jsonl.gz"
          for rep in range(1, 4)]
    for arm in ("official", "ours")
}


def read_rows(root, relative, manifest):
    path = root / relative
    raw = path.read_bytes()
    if relative.endswith(".gz"):
        raw = gzip.decompress(raw)
    manifest[relative] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def row_key(row):
    return row["dataset"], row["idx"], row["prompt_sha1"]


def check_serving(rows, name):
    assert Counter(r["dataset"] for r in rows) == TASK_COUNTS, name
    assert len({row_key(r) for r in rows}) == len(rows), name
    assert all(not r.get("error") and r["completion_tokens"] > 0
               and r["wall_s"] > 0 for r in rows), name
    assert all(0 <= r["ttft_s"] < r["wall_s"] for r in rows), name


def task_stats(rows):
    tokens = sum(r["completion_tokens"] for r in rows)
    wall = sum(r["wall_s"] for r in rows)
    verifies = sum(r.get("spec_verify_ct") or 0 for r in rows)
    return {
        "rows": len(rows), "completion_tokens": tokens,
        "verification_steps": verifies, "wall_seconds": wall,
        "tokens_per_second": tokens / wall,
        "accepted_length": tokens / verifies if verifies else None,
    }


def acceptance_evaluations(root, manifest):
    """Use deterministic T0 records and replicated T1 sampling runs."""
    payloads = []
    for name in ("official", "optimized"):
        relative = SERVING_SOURCES[0][name]
        rows = read_rows(root, relative, manifest)
        check_serving(rows, relative)
        assert all(r["spec_verify_ct"] > 0 for r in rows)
        payloads.append({"records": rows, "temperature": 0})
    base, arm = payloads
    assert {row_key(r) for r in base["records"]} == {row_key(r) for r in arm["records"]}
    baseline, ours = summarize(base), summarize(arm)
    results = {"0": {
        "source_files": list(SERVING_SOURCES[0].values()),
        "dspark": baseline, "ours": ours,
        "gain_percent": 100 * (ours["macro"] / baseline["macro"] - 1),
    }}

    summaries = {}
    identities = None
    for name, paths in T1_ACCEPTANCE_SOURCES.items():
        runs = []
        for relative in paths:
            rows = read_rows(root, relative, manifest)
            check_serving(rows, relative)
            assert all(r["spec_verify_ct"] > 0 for r in rows)
            keys = {row_key(r) for r in rows}
            identities = keys if identities is None else identities
            assert keys == identities
            runs.append(summarize({"records": rows, "temperature": 1}))
        macro_values = np.array([run["macro"] for run in runs])
        summaries[name] = {
            "n_records": 3030,
            "n_launches": len(runs),
            "per_task": {task: float(np.mean([run["per_task"][task] for run in runs]))
                         for task in TASK_COUNTS},
            "macro": float(macro_values.mean()),
            "macro_sd_across_launches": float(macro_values.std(ddof=1)),
            "macro_sem_across_launches": float(macro_values.std(ddof=1) / np.sqrt(len(runs))),
            "run_macros": macro_values.tolist(),
        }
    ratio = summaries["ours"]["macro"] / summaries["official"]["macro"]
    relative_sem = np.hypot(
        summaries["ours"]["macro_sem_across_launches"] / summaries["ours"]["macro"],
        summaries["official"]["macro_sem_across_launches"] / summaries["official"]["macro"],
    )
    results["1"] = {
        "source_files": T1_ACCEPTANCE_SOURCES,
        "dspark": summaries["official"], "ours": summaries["ours"],
        "gain_percent": 100 * (ratio - 1),
        "gain_sem_percentage_points": float(100 * ratio * relative_sem),
        "uncertainty": "Across-launch SD for each arm; first-order propagated 1 SEM for the ratio.",
    }
    return results


def serving_comparison(base, arm):
    b = {row_key(r): r for r in base}
    a = {row_key(r): r for r in arm}
    assert a.keys() == b.keys()
    assert all(a[k]["prompt_tokens"] == b[k]["prompt_tokens"] for k in b)
    assert all(r.get("spec_verify_ct", 0) > 0 for r in arm)
    tasks = {}
    for task in TASK_COUNTS:
        aa = [r for r in arm if r["dataset"] == task]
        bb = [r for r in base if r["dataset"] == task]
        sa, sb = task_stats(aa), task_stats(bb)
        tasks[task] = {
            **sa, "baseline_tokens_per_second": sb["tokens_per_second"],
            "speedup": sa["tokens_per_second"] / sb["tokens_per_second"],
        }
    return {
        "rows": len(arm), "prompt_clusters": len({(r["dataset"], r["prompt_sha1"])
                                                   for r in arm}),
        "output_hash_mismatches_vs_ar": sum(a[k]["sha1"] != b[k]["sha1"] for k in b),
        "per_task": tasks,
        "macro_accepted_length": float(np.mean([s["accepted_length"] for s in tasks.values()])),
        "macro_speedup": float(np.mean([s["speedup"] for s in tasks.values()])),
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=here / "historical_20260915")
    parser.add_argument("--out", type=Path, default=here / "generated")
    args = parser.parse_args()
    manifest = {}
    serving = {}
    for temp in (0, 1):
        base_path = f"ar/t{temp}.jsonl.gz"
        baseline = read_rows(args.data, base_path, manifest)
        check_serving(baseline, base_path)
        serving[str(temp)] = {}
        for name, relative in SERVING_SOURCES[temp].items():
            rows = read_rows(args.data, relative, manifest)
            check_serving(rows, relative)
            serving[str(temp)][name] = serving_comparison(baseline, rows)
    runtime_configs = {}
    for temp, sources in SERVING_SOURCES.items():
        for name, relative in sources.items():
            config_path = relative.replace(".jsonl.gz", "_config.json")
            if not (args.data / config_path).exists():
                continue            # the T1 official timing predates the config record
            raw = (args.data / config_path).read_bytes()
            manifest[config_path] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            config = json.loads(raw)
            assert config["temperature"] == temp
            assert config["max_tokens"] == 2048 and config["seed"] == 980406
            assert config["expected_requests"] == 3030
            assert config["git_revision"] == "8f51a8adf6db9e6aca844eb75028c1fcbaa47a2a"
            runtime_configs[f"{temp}/{name}"] = config
    report = {
        "serving_definition": "Within each task: total completion tokens / total client wall time; "
                              "speedup divides by the no-speculation AR rate; macro is the mean over nine tasks.",
        "timing_scope": "Sequential requests, including prefill and HTTP streaming overhead.",
        "serving_runtime": {"target": "Qwen/Qwen3-4B", "gpu": "NVIDIA H100 80GB HBM3",
                            "dtype": "bfloat16", "request_concurrency": 1, "speculative_slots": 7,
                            "ours": "optimized short convolution and precomputed attention-head projections; folded lattice enabled",
                            "git_revision": "8f51a8adf6db9e6aca844eb75028c1fcbaa47a2a",
                            "comparison_sources": SERVING_SOURCES,
                            "configs": runtime_configs},
        "serving": serving, "acceptance": acceptance_evaluations(args.data, manifest),
        "source_files": manifest,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "historical_20260915.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({t: {a: (v["macro_accepted_length"], v["macro_speedup"]) for a, v in d.items()}
                      for t, d in serving.items()}, indent=2))


if __name__ == "__main__":
    main()

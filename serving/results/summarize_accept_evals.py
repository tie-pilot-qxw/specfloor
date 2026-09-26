"""Recompute the training-trajectory and one-epoch component tables from per-request records.

    python summarize_accept_evals.py             # reads accept_evals/, writes generated/

Uses only saved evaluations (eval_accept/sglang_paired_accept.py output); never
launches a model. The trajectory is the 10-epoch run at temperature 1 on the
3,030 rows; the components are one-epoch checkpoints at temperature 0 on 430
rows (the first 50 of each task, all 30 of AIME25). The point estimate retains
every benchmark row. Bootstrap units are prompt hashes within each task: repeated
prompts are summed into one cluster, not silently overwritten or counted as
independent observations. Both arms use the same sampled clusters in each draw.
"""
import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TASKS = ("gsm8k", "math500", "aime25", "humaneval", "mbpp",
         "livecodebench", "mt-bench", "alpaca", "arena-hard-v2")


def read(root, name):
    payload = json.loads(gzip.decompress((root / f"{name}.json.gz").read_bytes()))
    rows = payload["records"]
    assert all(not r.get("error") and r.get("spec_verify_ct", 0) > 0
               and r.get("completion_tokens", 0) > 0 for r in rows), name
    assert set(r["dataset"] for r in rows) == set(TASKS), name
    return payload


def summarize(payload):
    per_task = {}
    for task in TASKS:
        rows = [r for r in payload["records"] if r["dataset"] == task]
        per_task[task] = sum(r["completion_tokens"] for r in rows) / sum(
            r["spec_verify_ct"] for r in rows)
    return {"n_records": len(payload["records"]), "per_task": per_task,
            "macro": sum(per_task.values()) / len(TASKS)}


def paired_bootstrap(base, arm, draws):
    for field in ("temperature", "max_new_tokens", "seed", "cap", "official_subset"):
        assert base.get(field) == arm.get(field), field
    rng = np.random.default_rng(20260909)
    boot = np.zeros((draws, 2))
    total_clusters = 0
    for task in TASKS:
        clusters = []
        memberships = []
        for payload in (base, arm):
            grouped = defaultdict(lambda: np.zeros(2))
            members = defaultdict(list)
            for r in payload["records"]:
                if r["dataset"] == task:
                    grouped[r["prompt_sha1"]] += (r["completion_tokens"], r["spec_verify_ct"])
                    members[r["prompt_sha1"]].append(r["idx"])
            clusters.append(grouped)
            memberships.append({k: sorted(v) for k, v in members.items()})
        assert memberships[0] == memberships[1], task
        keys = sorted(clusters[0])
        total_clusters += len(keys)
        arrays = [np.array([c[k] for k in keys]) for c in clusters]
        for start in range(0, draws, 100):
            size = min(100, draws - start)
            ids = rng.integers(len(keys), size=(size, len(keys)))
            for i, array in enumerate(arrays):
                sums = array[ids].sum(axis=1)
                boot[start:start + size, i] += sums[:, 0] / sums[:, 1] / len(TASKS)
    return {"draws": draws, "seed": 20260909, "prompt_clusters": total_clusters,
            "delta_ci95": np.quantile(boot[:, 1] - boot[:, 0], [.025, .975]).tolist(),
            "relative_gain_pct_ci95": np.quantile(
                100 * (boot[:, 1] / boot[:, 0] - 1), [.025, .975]).tolist()}


COMPONENTS = [("lat_vanilla", "Vanilla DSpark"), ("lattice_dflash2_repro", "DFlash2 reproduction"),
              ("lat_shortconv", "Vanilla + our short convolution"),
              ("lat_slotembed", "Vanilla + slot embeddings"),
              ("lat_attnhead", "Prefix-attention head"),
              ("lat_attnconv", "Head + our convolution + slot embeddings")]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, default=here / "accept_evals")
    parser.add_argument("--out", type=Path, default=here / "generated")
    parser.add_argument("--draws", type=int, default=4000)
    args = parser.parse_args()
    base = read(args.eval_root, "paper_official")
    arm = read(args.eval_root, "paper_10ep_final")
    b, a = summarize(base), summarize(arm)
    report = {"baseline": b, "final": a,
              "delta": a["macro"] - b["macro"],
              "relative_gain_pct": 100 * (a["macro"] / b["macro"] - 1),
              "bootstrap": paired_bootstrap(base, arm, args.draws)}
    report["trajectory"] = {str(ep): summarize(read(args.eval_root, (
        f"paper_10ep_ep{ep}" if ep < 10 else "paper_10ep_final")))["macro"]
        for ep in range(1, 11)}
    variants = {name: summarize(read(args.eval_root, name))["macro"] for name, _ in COMPONENTS}
    report["one_epoch_design_variants_temperature0"] = variants
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "accept_evals.json").write_text(json.dumps(report, indent=2) + "\n")
    header = "% Generated by serving/results/summarize_accept_evals.py; table body rows only."
    (args.out / "solution_trajectory.tex").write_text("\n".join([
        header, " & ".join(f"{report['trajectory'][str(ep)]:.2f}" for ep in range(1, 11)) + r" \\", ""]))
    vanilla = variants["lat_vanilla"]
    rows = [f"{label} & {variants[name]:.2f} & " +
            ("---" if name == "lat_vanilla" else f"${variants[name] - vanilla:+.2f}$") + r" \\"
            for name, label in COMPONENTS]
    (args.out / "solution_components.tex").write_text("\n".join([header, *rows, ""]))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

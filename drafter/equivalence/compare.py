"""Compare two run_step.py results bit for bit.

    python drafter/equivalence/compare.py reference.json candidate.json [--json out.json]

Exits non-zero if anything differs.  Metrics that exist on only one side are
listed separately (the release drops research-only diagnostics); metrics present
on both sides must be bit-identical.
"""

import argparse
import json
import sys


def _cmp_dict(a, b):
    keys = sorted(set(a) | set(b))
    only_a = [k for k in keys if k not in b]
    only_b = [k for k in keys if k not in a]
    diff = [k for k in keys if k in a and k in b and a[k] != b[k]]
    return only_a, only_b, diff


def compare(ref, cand):
    report = {"equal": True, "sections": {}}

    def section(name, only_a, only_b, diff, n):
        ok = not (only_a or only_b or diff)
        report["sections"][name] = {"compared": n, "only_reference": only_a,
                                    "only_candidate": only_b, "different": diff,
                                    "equal": ok}
        report["equal"] &= ok

    for key in ("init", "init_buffers"):
        a, b = ref[key], cand[key]
        section(key, *_cmp_dict(a, b), len(set(a) & set(b)))
    t_a, t_b = set(ref["trainable"]), set(cand["trainable"])
    section("trainable", sorted(t_a - t_b), sorted(t_b - t_a), [], len(t_a & t_b))
    if "load" in ref or "load" in cand:
        la, lb = ref.get("load", {}), cand.get("load", {})
        diff = [k for k in ("missing", "unexpected", "tensors") if la.get(k) != lb.get(k)]
        section("load", [], [], diff, 3)
        report["sections"]["load"]["strict_clean"] = [
            not (la.get("missing") or la.get("unexpected")),
            not (lb.get("missing") or lb.get("unexpected"))]
    metric_only = {}
    for run in ("run_init", "run_checkpoint"):
        if run not in ref and run not in cand:
            continue
        ra, rb = ref.get(run), cand.get(run)
        if ra is None or rb is None:
            section(run, [run] if ra else [], [run] if rb else [], [], 0)
            continue
        diffs, n = [], 0
        if len(ra["steps"]) != len(rb["steps"]):
            diffs.append("number of steps")
        for i, (sa, sb) in enumerate(zip(ra["steps"], rb["steps"])):
            for k in sorted(set(sa) | set(sb)):
                if k in ("loss",):
                    continue
                if k == "metrics":
                    oa, ob, d = _cmp_dict(sa[k], sb[k])
                    diffs += [f"step{i}.metrics.{m}" for m in d]
                    n += len(set(sa[k]) & set(sb[k]))
                    metric_only.setdefault("only_reference", set()).update(oa)
                    metric_only.setdefault("only_candidate", set()).update(ob)
                    continue
                n += 1
                if sa.get(k) != sb.get(k):
                    diffs.append(f"step{i}.{k}")
        oa, ob, d = _cmp_dict(ra["grads"], rb["grads"])
        diffs += [f"grad.{k}" for k in d + oa + ob]
        n += len(ra["grads"])
        if ra["grad_norm_bits"] != rb["grad_norm_bits"]:
            diffs.append("grad_norm")
        section(run, [], [], diffs, n)
        report["sections"][run]["losses"] = [
            [s["loss"] for s in ra["steps"]], [s["loss"] for s in rb["steps"]]]
        report["sections"][run]["grad_norm"] = [ra["grad_norm"], rb["grad_norm"]]
    report["metrics_only_on_one_side"] = {k: sorted(v) for k, v in metric_only.items()}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ref = json.load(open(args.reference))
    cand = json.load(open(args.candidate))
    report = compare(ref, cand)
    report["reference"], report["candidate"] = args.reference, args.candidate
    for name, s in report["sections"].items():
        status = "EQUAL" if s["equal"] else "DIFFERENT"
        extra = ""
        if "losses" in s:
            extra = f"  losses={s['losses'][1]}  grad_norm={s['grad_norm'][1]:.6g}"
        print(f"{name:16s} {status:9s} ({s['compared']} compared){extra}")
        for k in ("only_reference", "only_candidate", "different"):
            if s[k]:
                print(f"    {k}: {s[k][:8]}{' ...' if len(s[k]) > 8 else ''}")
    for k, v in report["metrics_only_on_one_side"].items():
        if v:
            print(f"metrics {k}: {v}")
    print("RESULT:", "bit-identical" if report["equal"] else "DIFFERENT")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    sys.exit(0 if report["equal"] else 1)


if __name__ == "__main__":
    main()

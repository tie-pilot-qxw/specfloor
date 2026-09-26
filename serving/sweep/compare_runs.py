"""Compare three run_arm.py outputs at one temperature, with the paper's aggregation.

    python compare_runs.py --baseline runs/t0_ar --official runs/t0_official --ours runs/t0_ours

At each concurrency C, within each task: throughput is output tokens over the
task's elapsed time, speedup divides by AR's throughput, and accepted length is
output tokens over verification steps. Macro values average the nine tasks
equally. The accepted length reported in the main table averages each task over
the five concurrencies first, then the nine tasks. This is the aggregation of
results/summarize_serving_sweep.py, applied to fresh runs instead of the archive.
"""
import argparse
import json
from pathlib import Path
import statistics

from common import TASK_COUNTS


def load(run, arm):
    points = {}
    for path in sorted(Path(run).glob(f"{arm}.c*.summary.json")):
        c = int(path.name.split(".")[1][1:])
        if not (Path(run) / f"{arm}.c{c}.complete.json").exists():
            continue
        points[c] = json.loads(path.read_text())
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--official", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--json", type=Path, help="also write the comparison here")
    args = ap.parse_args()
    runs = {a: load(getattr(args, a), a) for a in ("baseline", "official", "ours")}
    common = sorted(set.intersection(*(set(p) for p in runs.values())))
    if not common:
        raise SystemExit("No concurrency is complete in all three runs")
    tasks = list(TASK_COUNTS)
    report = {}
    print(f"{'C':>3} {'S DSpark':>9} {'S Ours':>8} {'gain':>8} {'tau DSpark':>11} {'tau Ours':>9}")
    for c in common:
        rate = {a: {n: runs[a][c][n]["system_output_tokens_per_second"] for n in tasks} for a in runs}
        speed = {a: statistics.mean(rate[a][n] / rate["baseline"][n] for n in tasks)
                 for a in ("official", "ours")}
        gain = statistics.mean(rate["ours"][n] / rate["official"][n] for n in tasks)
        tau = {a: statistics.mean(runs[a][c][n]["accepted_length"] for n in tasks)
               for a in ("official", "ours")}
        report[str(c)] = dict(macro_speedup_over_ar=speed, mean_task_ours_over_official=gain,
                              macro_accepted_length=tau)
        print(f"{c:>3} {speed['official']:>9.3f} {speed['ours']:>8.3f} {100 * (gain - 1):>+7.2f}% "
              f"{tau['official']:>11.3f} {tau['ours']:>9.3f}")
    if len(common) == 5:
        for a in ("official", "ours"):
            per_task = {n: statistics.mean(runs[a][c][n]["accepted_length"] for c in common) for n in tasks}
            report[f"tau_five_concurrency_{a}"] = statistics.mean(per_task.values())
        print(f"five-concurrency tau: DSpark {report['tau_five_concurrency_official']:.2f}, "
              f"ours {report['tau_five_concurrency_ours']:.2f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

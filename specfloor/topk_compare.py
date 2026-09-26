"""Paired top-20 vs top-256 calibration for the frontier-scale floor.

The V4-Pro floor is read through an API that caps top_logprobs at 20, so every
p_Z there is a top-20 truncation renormalised over the reported masses. Locally
the SAME anchors can be read both ways, which turns the coarsening into a
controlled comparison.

The comparison has to be made cell by cell, not column by column. A top-20 read
fails the residual gate far more often, so the two columns as printed describe
different sub-populations, and their difference mixes the coarsening with a
change in which cells survive. Reported here:

  (1) the paired difference on cells passing the gate under BOTH reads --
      does truncation move the number it can compute?
  (2) gate survival under each read -- how many cells does it cost?
  (3) the floor of the cells top-20 drops and top-256 keeps -- is the loss
      random, or does it fall on exactly the cells with the largest floors?

(3) is the one that matters. A truncation that biased the value would show up
in (1) and be correctable; a truncation that silently deletes the high-floor
cells is a selection effect, invisible in the point estimate.
"""
import argparse
import glob
import json
import random

from specfloor.records import open_text

GATE = 1e-3


def load(pattern):
    out = {}
    for f in glob.glob(pattern):
        dom = f.split("/")[-1].split(".")[0]
        for line in open_text(f):
            r = json.loads(line)
            out[(dom, r["prompt_id"], r["t"])] = r
    return out


def ok(row, m, k, gate=GATE):
    """Cell present, finite, and inside the residual gate."""
    t = row["T"].get(m, {}).get(str(k))
    r = row["resid"].get(m, {}).get(str(k))
    return t is not None and r is not None and r <= gate


def boot_ci(vals, keys, B=4000, seed=20260820):
    """Cluster bootstrap over prompts."""
    by = {}
    for v, kk in zip(vals, keys):
        by.setdefault(kk, []).append(v)
    ks = list(by)
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        s = []
        for _ in range(len(ks)):
            s += by[ks[rng.randrange(len(ks))]]
        if s:
            draws.append(sum(s) / len(s))
    draws.sort()
    q = lambda p: draws[max(0, min(len(draws) - 1, int(p * len(draws))))]
    return q(0.025), q(0.975)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Paired comparison of two vocabulary truncations on the same anchors.")
    ap.add_argument("--narrow", required=True,
                    help="glob for the narrower read (e.g. the top-20 arm)")
    ap.add_argument("--wide", required=True,
                    help="glob for the wider read on the SAME anchors")
    ap.add_argument("--gate", type=float, default=GATE,
                    help="residual gate; a cell counts only if it passes under BOTH reads")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    A, B = load(args.narrow), load(args.wide)
    if not A or not B:
        raise SystemExit(f"no records matched {args.narrow!r} / {args.wide!r}")
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit(
            "the two globs share no (domain, prompt, position) anchor. This "
            "comparison is only meaningful when both reads cover the same "
            "anchors -- rerun the narrow arm with the wide arm's corpus files, "
            "anchor set, M and seeds, changing only --top-k.")
    print(f"anchors: narrow {len(A)}  wide {len(B)}  common {len(common)}\n")

    for m in ("0", "1"):
        print(f"=== order {m}")
        print("%4s | %5s %9s %9s %11s | %13s %13s | %6s %9s"
              % ("slot", "n", "narrow", "wide", "paired diff",
                 "gate narr", "gate wide", "n_lost", "lost wide T"))
        for k in range(0, 7):
            both, keys = [], []
            for key in common:
                if ok(A[key], m, k, args.gate) and ok(B[key], m, k, args.gate):
                    both.append((A[key]["T"][m][str(k)], B[key]["T"][m][str(k)]))
                    keys.append(key[1])
            g20 = sum(1 for key in common if ok(A[key], m, k, args.gate))
            g256 = sum(1 for key in common if ok(B[key], m, k, args.gate))
            # cells the WIDE read resolves and the narrow one cannot: their floors
            lost = [B[key]["T"][m][str(k)] for key in common
                    if ok(B[key], m, k, args.gate) and not ok(A[key], m, k, args.gate)]
            if len(both) < 20:
                continue
            n = len(both)
            u = sum(x for x, _ in both) / n
            v = sum(y for _, y in both) / n
            d = [x - y for x, y in both]
            lo, hi = boot_ci(d, keys, args.boot, args.seed)
            lm = (sum(lost) / len(lost)) if lost else float("nan")
            print("%4d | %5d %9.4f %9.4f %+11.6f | %6d/%-6d %6d/%-6d | %6d %9.4f"
                  % (k, n, u, v, u - v, g20, len(common), g256, len(common),
                     len(lost), lm))
            print("%4s |       paired diff 95%% CI [%+.6f, %+.6f]" % ("", lo, hi))
        print()


if __name__ == "__main__":
    main()

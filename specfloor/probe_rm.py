"""R_m probe: how much of the missing path information do the last m tokens buy?

    R_m = (CE_B - CE_M_m) / (CE_B - CE_A)

CE_M_m conditions on a SAMPLED early segment plus the TRUE last m tokens, then
marginalises the early segment -- same log-of-mean convention as CE_B.

Cost note, and the reason this is a separate binary from probe_cheap: each anchor
needs ~K x |RM_ORDERS| x MX scored prefills. Under transformers each of those was
a full re-prefill of the whole prefix; under sglang the prefix sits in the radix
cache and only the <= gamma divergent tokens are new work, which is what makes
the pass affordable at all. The budget is nonetheless controlled by ANCHOR
SUBSAMPLING -- run this only on informative anchors identified by the cheap pass,
stratified, per PROTOCOL.md sec.3.

Usage:
  python -m specfloor.probe_rm --corpus C0 --corpus-file runs/C0/gsm8k.jsonl \
      --cheap runs/C0/gsm8k.cheap.jsonl --out runs/C0/gsm8k.rm.jsonl
"""

from __future__ import annotations

# must precede any sglang import (sets SGLANG_RETURN_ORIGINAL_LOGPROB)
from specfloor import backend as B
from specfloor import backend_api as BA

import argparse
import collections
import json
import math
import pathlib
import random


from transformers import AutoTokenizer

from specfloor import config as C
from specfloor.corpus import resolve_stops
from specfloor.probe_cheap import (anchor_seed, ce_b_jackknife,
                                     sample_paths, score_paths)


def select_informative(cheap, budget, slot, seed=C.SEED,
                       eps=C.RM_ELIGIBILITY_THRESHOLD):
    """Stratified subsample of the R_m-ELIGIBLE anchors.

    Note the threshold: eps_R (0.05), not eps_info (0.01). R_m is a normalised
    ratio and a near-zero denominator makes it meaningless, so anchors between
    the two thresholds are informative for Part I but cannot support a stable
    ratio. Selecting them anyway is how the pilot spent 54% of the expensive
    pass producing `None`.
    """
    inf = [r for r in cheap if r["dCE"][slot] > eps]
    by = collections.defaultdict(list)
    for r in inf:
        by[r["stratum"]].append(r)
    rng = random.Random(seed)
    cells = sorted(by)
    if not cells:
        return [], {}
    per = max(1, budget // len(cells))
    out, taken = [], {}
    for c in cells:
        rng.shuffle(by[c])
        out += by[c][:per]
        taken[c] = min(per, len(by[c]))
    return out, taken


def ce_mixed_all(eng, prefix, gt, drawn, K, orders, slot_range):
    """CE_M_m for every (m, slot), from sampled early segment + true last-m.

    For slot k and order m the conditioning state is

        prefix + path[:k-m] + gt[k-m:k]

    and we want p(gt[k] | that). Scoring is done by appending gt[k] and reading
    the last teacher-forced NLL, so the estimand is identical to CE_A's and there
    is no second convention to keep in sync.

    All (m, k, path) requests go out in ONE batched call: they share `prefix`, so
    the radix cache prefills it once and each request only pays for its own short
    divergent tail.
    """
    P = len(prefix)
    seqs, tags, starts = [], [], []
    for m in orders:
        for k in slot_range:
            if m >= k:            # empty early segment => CE_M == CE_A, so R == 1
                continue          # by construction, not by measurement. Skip.
            for path in drawn:
                s = (list(prefix) + list(path[: k - m])
                     + list(gt[k - m: k]) + [gt[k]])
                seqs.append(s)
                tags.append((m, k))
                # Score the m REVEALED tokens as well as gt[k]. The revealed
                # block sits at positions len(s)-1-m .. len(s)-2, so this start
                # returns exactly m+1 values: log w(s) for the reveal, then
                # log f(s) for gt[k]. It is the same forward pass either way.
                starts.append(len(s) - 1 - m)
    if not seqs:
        return ({m: {k: None for k in slot_range} for m in orders},
                {m: {k: None for k in slot_range} for m in orders},
                {m: {k: None for k in slot_range} for m in orders})

    nll = eng.teacher_forced_nll(seqs, starts)
    by = collections.defaultdict(list)
    for (m, k), v in zip(tags, nll):
        # v[:m] are the revealed tokens under THIS sampled prefix; v[m] is gt[k].
        logw = -sum(v[:m]) if m else 0.0
        by[(m, k)].append((logw, math.exp(-v[m])))

    snis, interv, ess = ({m: {} for m in orders}, {m: {} for m in orders},
                         {m: {} for m in orders})
    for m in orders:
        for k in slot_range:
            rows = by.get((m, k))
            if not rows:
                snis[m][k] = interv[m][k] = ess[m][k] = None
                continue
            fs = [f for _, f in rows]
            # INTERVENTIONAL: force the reveal onto prior-sampled prefixes and
            # average. This is what every earlier version of this probe reported.
            _, interv[m][k], _, _ = ce_b_jackknife(fs)
            # CONDITIONAL: the estimand R_m actually names. Reweight the prior
            # prefixes by how likely each was to have produced the revealed
            # tokens -- self-normalised importance sampling, exact in the limit:
            #     p(y_k | X, z*) = E_s[w f] / E_s[w],  w(s) = p(z* | X, s)
            mx = max(lw for lw, _ in rows)          # stabilise before exp
            ws = [math.exp(lw - mx) for lw, _ in rows]
            sw = sum(ws)
            snis[m][k] = (-math.log(max(sum(w * f for w, (_, f) in zip(ws, rows))
                                        / sw, 1e-12))
                          if sw > 0 else None)
            # Effective sample size: SNIS collapses exactly when the revealed
            # suffix is improbable under most sampled prefixes, which is the
            # regime that matters. Gated by the caller, never silently averaged.
            ess[m][k] = (sw * sw) / sum(w * w for w in ws) if sw > 0 else 0.0
    return snis, interv, ess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(C.CORPORA))
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--budget", type=int, default=C.RM_ANCHORS_PER_DOMAIN)
    ap.add_argument("--mixed-paths", type=int, default=C.RM_MIXED_PATHS)
    ap.add_argument("--slot", type=int, default=C.GAMMA - 1,
                    help="slot used to define the informative set")
    B.add_engine_args(ap)
    BA.add_api_args(ap)
    args = ap.parse_args()

    K = C.GAMMA
    policy = C.CORPORA[args.corpus]

    seqs = {}
    with open(args.corpus_file) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if d.get("corpus") != args.corpus:
                    raise SystemExit(
                        f"corpus mismatch: --corpus {args.corpus} but "
                        f"{args.corpus_file} contains {d.get('corpus')}")
                seqs[d["prompt_id"]] = d
    cheap = [json.loads(l) for l in open(args.cheap) if l.strip()]
    bad = {r.get("corpus") for r in cheap} - {args.corpus}
    if bad:
        raise SystemExit(f"cheap-probe file is from corpus {sorted(bad)}, "
                         f"not {args.corpus}")

    chosen, taken = select_informative(cheap, args.budget, args.slot)
    n_info = sum(1 for r in cheap if r["dCE"][args.slot] > C.INFORMATIVE_THRESHOLD)
    n_elig = sum(1 for r in cheap if r["dCE"][args.slot] > C.RM_ELIGIBILITY_THRESHOLD)
    print(f"anchors informative for Part I (dCE > {C.INFORMATIVE_THRESHOLD}): "
          f"{n_info} / {len(cheap)}")
    print(f"anchors ELIGIBLE for R_m    (dCE > {C.RM_ELIGIBILITY_THRESHOLD}): "
          f"{n_elig} / {len(cheap)}   <-- the expensive pass runs on these only")
    print(f"selected {len(chosen)} for the expensive pass:")
    for c in sorted(taken):
        flag = "" if taken[c] >= C.MIN_INFORMATIVE_PER_CELL else "   <-- exploratory"
        print(f"  {c:<26} {taken[c]:>5}{flag}")
    if not chosen:
        raise SystemExit(
            f"no anchors with dCE > eps_R={C.RM_ELIGIBILITY_THRESHOLD} -- nothing "
            f"to do. R_m is undefined on this cell, which is a result, not an error.")

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)

    max_ctx = max(seqs[a["prompt_id"]]["prompt_len"] + a["t"] + K
                  for a in chosen) + 8

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    args.context_length = args.context_length or max_ctx
    with BA.make_target(args, need_logprobs=True) as eng, out.open("w") as fout:
        for i, a in enumerate(chosen):
            seq = seqs[a["prompt_id"]]
            full = seq["prompt_ids"] + seq["response_ids"]
            cut = seq["prompt_len"] + a["t"]
            prefix, gt = full[:cut], full[cut: cut + K]
            if len(gt) < K:
                continue

            # warm the radix tree before fanning out (see backend.warm_prefix)
            eng.warm_prefix(prefix)

            # SAMPLE SPLITTING. The paths here are drawn with an offset seed, so
            # they are independent of the draws whose CE_B selected this anchor.
            # Reusing the cheap pass's CE_B would be selection-on-noise: anchors
            # are chosen because their dCE estimate came out high, and that same
            # upward-biased estimate would then sit in the ratio. Worse, CE_B
            # appears in BOTH numerator and denominator, so an error d moves
            # R = (N+d)/(D+d) toward 1 -- straight toward the H2 conclusion.
            sb = anchor_seed(a["prompt_id"], a["t"],
                             C.SEED ^ C.RM_RESCORE_SEED_OFFSET)
            drawn, _ = sample_paths(eng, prefix, args.mixed_paths, policy, K,
                                    stop_ids, sb)

            # CE_B recomputed on THESE paths. Sharing the draws with CE_M is
            # deliberate (common random numbers): the numerator CE_B - CE_M is a
            # difference of two quantities with correlated MC error, so it is far
            # better conditioned than two independent estimates would be.
            probs = score_paths(eng, prefix, gt, drawn, K)
            ce_b_fresh = [ce_b_jackknife([probs[m][k] for m in range(len(probs))])[1]
                          for k in range(K)]
            ce_a = a["ce_A"]          # exact, teacher-forced: no MC, no bias

            cm, cm_iv, cm_ess = ce_mixed_all(
                eng, prefix, gt, drawn, K, C.RM_ORDERS, range(1, K))

            rec = {**{q: a[q] for q in ("prompt_id", "t", "context", "stratum", "pi")},
                   "corpus": args.corpus, "ce_A": ce_a,
                   "ce_B": ce_b_fresh,            # independent of the selection
                   "ce_B_select": a["ce_B"],      # kept to audit the split
                   "dCE_select": a["dCE"],
                   "mixed_paths": args.mixed_paths,
                   "ce_M": {}, "R": {}}
            # numerator/denominator kept separately so stats.py can form a
            # RATIO OF MEANS across anchors instead of a mean of per-anchor
            # ratios -- the latter is what blows up when a denominator is small.
            rec["den"] = {str(k): ce_b_fresh[k] - ce_a[k] for k in range(1, K)}
            # ce_M is the CONDITIONAL (SNIS) estimate -- the estimand R_m names.
            # ce_M_interv is the interventional one every earlier run reported;
            # both are kept so the size of the correction is auditable rather
            # than asserted. ess gates the SNIS column.
            rec["ce_M_interv"] = {str(m): {str(k): cm_iv[m].get(k)
                                           for k in range(1, K)}
                                  for m in C.RM_ORDERS}
            rec["ess"] = {str(m): {str(k): cm_ess[m].get(k) for k in range(1, K)}
                          for m in C.RM_ORDERS}
            for m in C.RM_ORDERS:
                rec["ce_M"][str(m)] = {str(k): cm[m].get(k) for k in range(1, K)}
                rec["num"] = rec.get("num", {})
                rec["num"][str(m)] = {
                    str(k): (ce_b_fresh[k] - cm[m][k]
                             if cm[m].get(k) is not None else None)
                    for k in range(1, K)
                }
                rec["R"][str(m)] = {
                    str(k): ((ce_b_fresh[k] - cm[m][k]) / d
                             if cm[m].get(k) is not None
                             and (d := ce_b_fresh[k] - ce_a[k]) > C.RM_ELIGIBILITY_THRESHOLD
                             else None)
                    for k in range(1, K)
                }
            fout.write(json.dumps(rec) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(chosen)}", flush=True)

    print(f"== R_m written to {out}", flush=True)
    print(f"   selection used eps_R={C.RM_ELIGIBILITY_THRESHOLD} on the cheap "
          f"pass; CE_B was RE-ESTIMATED here on independent draws, so the "
          f"selection\n   does not leak into the ratio.")
    print(f"   mixed paths = {args.mixed_paths}; CE_M shares those draws with "
          f"CE_B (common random numbers) and uses the same jackknife "
          f"correction,\n   so the numerator is a well-conditioned difference.")
    print( "   stats.py reports the RATIO OF MEANS as the headline; per-anchor "
           "R is kept for\n   diagnostics only.")


if __name__ == "__main__":
    main()

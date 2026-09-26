"""Cheap probe: CE_A, CE_B, CE_commit, dCE -- with the frozen M ladder.

Convention, stated once and applied everywhere:
  * ROLLOUT POLICY = THE CORPUS'S OWN POLICY, including where it STOPS. A path
    that samples a stop token is frozen there; continuing past it would condition
    on prefix+EOS+junk, a state the generative process never produces, and would
    collapse p(y_k) exactly in the late-position bin.
  * PROBABILITIES ARE ALWAYS READ FROM THE RAW HEAD -- no temperature, no
    truncation. We are measuring p, not the sampling process.

Quantities per slot k:
  CE_A      = -log p(y_k | prefix, TRUE path)                  teacher forced, exact
  CE_B      = -log mean_m p(y_k | prefix, path_m)              LOG OF THE MEAN
  CE_commit = mean_m [ -log p(y_k | prefix, path_m) ]          MEAN OF THE LOG
  dCE       = CE_B - CE_A

CE_commit >= CE_B by Jensen; the gap is across-path heterogeneity of the
predictive, NOT a chain drafter's regret. Do not label it as one.

CE_B is a log-of-a-mean, so the plug-in estimator carries an upward MC bias of
about CV^2/(2M) nats, largest at exactly the heterogeneous anchors that carry the
story -- and it would inflate P(dCE > eps), the headline incidence. That is what
the jackknife correction is for; `ce_B` is the corrected value and `ce_B_plugin`
is kept so the correction's size stays auditable.

M IS FIXED AT MAIN_M. There is no per-anchor SE target, because SE is not a
meaningful stopping rule here: the required M is heavy-tailed to the point of
absurdity (pilot: median 0, p75 129, p90 855, max 19537 to reach SE < 0.05), and
the paper's claims are population-level. The one adaptive rule kept is boundary
escalation -- extra paths for anchors whose dCE could cross eps_info under MC
noise. Everything else rides on the M ladder showing that the AGGREGATE
statistics do not move between M = 32/64/128/256.

A thresholded statistic is not automatically safe just because the mean is:
E[1(dCE_hat > eps)] != P(dCE > eps) unless the noise is small WHERE THE THRESHOLD
IS. So `ambiguous_slots` records, per anchor, the slots still within
AMBIGUITY_K*SE of eps_info, and stats.py quotes that count next to every
incidence figure. In the pilot it was 0 of 65 high-SE slots -- the noise and the
threshold lived in disjoint regions -- but that is a measured property of the
workload, not a theorem, so it is checked every run rather than assumed.

-------------------------------------------------------------------------------
SAMPLING AND SCORING ARE SEPARATE PASSES.

Pass 1 samples M continuations under the corpus policy. Pass 2 re-reads each
path teacher-forced and pulls p(gt[k]) out of the RAW head. Doing it in one pass
-- reading the sampler's own output logprobs -- would have been cheaper and
wrong: sglang's decode path divides the logits by the temperature IN PLACE
(srt/layers/sampler.py:189) before computing logprobs, so under C1
(T=0.7/top_p=0.8/top_k=20) every probability would have come back rescaled.
backend.py sets SGLANG_RETURN_ORIGINAL_LOGPROB=1 as a second line of defence,
but the input-side read is raw by construction and does not depend on the flag.

Pass 2 is cheap despite being a second forward pass: the prefix is already in
sglang's radix cache from pass 1, so only the gamma path tokens are new work.
-------------------------------------------------------------------------------

Usage:
  python -m specfloor.probe_cheap --corpus C0 \\
      --corpus-file runs/C0/gsm8k.jsonl --anchors runs/C0/gsm8k.anchors.jsonl \\
      --out runs/C0/gsm8k.cheap.jsonl
"""

from __future__ import annotations

# must precede any sglang import (sets SGLANG_RETURN_ORIGINAL_LOGPROB)
from specfloor import backend as B
from specfloor import backend_api as BA

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random


from transformers import AutoTokenizer

from specfloor import config as C
from specfloor.corpus import resolve_stops


def anchor_seed(prompt_id: str, t: int, seed: int) -> int:
    """Stable per-anchor RNG seed. Reproducible across processes and machines,
    which builtin hash() is not."""
    h = hashlib.sha256(f"{prompt_id}\x00{t}\x00{seed}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# ----------------------------------------------------------------- stats ----
def ce_b_jackknife(pg) -> tuple[float, float, float, bool]:
    """Returns (plugin, corrected, SE, jk_ok) for -log(mean p).

    `pg` is a sequence of per-path probabilities of the same ground-truth token.

    VALIDITY GUARD. The jackknife extrapolation n*plug - (n-1)*mean(loo) assumes
    the bias is O(1/n) and smooth. That assumption fails exactly when one path
    dominates the mean: dropping that path collapses pbar by ~n, its
    leave-one-out term explodes, and the extrapolation overshoots. Concretely,
    for pg = [0.9, 1e-9 x 7] the plug-in is 2.18 nats and the raw jackknife is
    -13.22 -- a NEGATIVE cross-entropy, which is impossible since
    CE = -log(pbar) with pbar <= 1.

    The admissible range is [0, plug]: the MC bias in log-of-a-mean is upward, so
    a correction must reduce the estimate, and CE cannot go below zero. Outside
    that range the expansion has failed and the plug-in is returned instead, with
    jk_ok=False so the count is reportable rather than silent.

    This is not a cosmetic clamp. An anchor that trips it has an effective sample
    size near 1, and its SE (returned unchanged) is correspondingly enormous --
    the downstream ambiguity gate is what decides whether that matters.
    """
    pg = [float(x) for x in pg]
    n = len(pg)
    if not n:
        return float("nan"), float("nan"), float("inf"), False
    tot = sum(pg)
    plug = -math.log(max(tot / n, 1e-12))
    pbar = tot / n
    if n < 3:
        return plug, plug, float("inf"), False
    var = sum((x - pbar) ** 2 for x in pg) / (n - 1)
    se = math.sqrt(var / n) / pbar if pbar > 0 else float("inf")

    loo = [-math.log(max((tot - x) / (n - 1), 1e-12)) for x in pg]
    corrected = n * plug - (n - 1) * (sum(loo) / n)
    jk_ok = 0.0 <= corrected <= plug + 1e-12
    return plug, (corrected if jk_ok else plug), se, jk_ok


def needs_escalation(pg_by_slot, ce_a) -> bool:
    """Boundary rule ONLY: escalate where extra paths can change the answer.

    The blanket `se > SE_TARGET` trigger is gone. It was the wrong stopping rule:
    se here is the coefficient of variation of the per-path probability divided
    by sqrt(M), and for the heterogeneous anchors that carry the story the CV is
    large enough that no feasible M reaches 0.05 (pilot: p90 needed M=855, max
    19537). Chasing it spent the budget on anchors nothing could fix while
    changing no published number.

    What remains is the trigger that buys something: anchors whose dCE sits close
    enough to eps_info that MC noise could flip their informative/not
    classification. Those are cheap (3 anchors in the pilot vs 12 for the SE
    floor) and they are precisely what keeps the noise away from the threshold.
    """
    for k, pg in enumerate(pg_by_slot):
        _, ce_b, se, _ = ce_b_jackknife(pg)
        if se != se or se == float("inf"):        # unusable SE -> cannot judge
            continue
        if abs((ce_b - ce_a[k]) - C.INFORMATIVE_THRESHOLD) < C.BOUNDARY_K * se:
            return True
    return False


def ambiguous_slots(dce, se, eps=None, k=None):
    """Slots where MC noise could flip the eps threshold. Reported, not dropped."""
    eps = C.INFORMATIVE_THRESHOLD if eps is None else eps
    k = C.AMBIGUITY_K if k is None else k
    return [i for i, (d, s) in enumerate(zip(dce, se))
            if s == s and s != float("inf") and abs(d - eps) < k * s]


# ------------------------------------------------------------- the probe ----
def sample_paths(eng, prefix, n, policy, K, stop_ids, seed_base):
    """M continuations of length K-1 under the corpus policy.

    A path that hits a stop token comes back short; it is held on stop_ids[0]
    for the remaining slots, which is the frozen convention -- the alternative
    (letting it run free past EOS) conditions on states the generative process
    never produces.
    """
    res = eng.generate_ids([list(prefix)] * n, policy, K - 1, stop_ids,
                           seed=[seed_base + i for i in range(n)])
    paths, nlive = [], []
    for p, _finish in res:
        p = p[: K - 1]
        # How many REAL tokens this path produced. Slot k reads
        # p(gt[k] | prefix, path[:k]) and needs k real tokens, so this path is
        # admissible at slots 0..nlive and at no slot beyond.
        nlive.append(len(p))
        # The tail is still padded, because the scoring call wants one shape for
        # every sequence -- but the padded rows are DROPPED per slot rather than
        # averaged in. Scoring them would query the model after
        # prefix + EOS + EOS ..., which is neither the rollout law (the path is
        # over) nor an absorbing state (the model happily continues), and the
        # full-vocabulary rollout in probe_rpre removes those paths outright, so
        # keeping them here would have the two implementations measuring
        # different post-EOS populations.
        p += [stop_ids[0]] * (K - 1 - len(p))
        paths.append(p)
    return paths, nlive


def score_paths(eng, prefix, gt, paths, K):
    """p(gt[k] | prefix, path[:k]) for every path and every slot k in 0..K-1.

    Index contract, spelled out because an off-by-one here is invisible:
      seq   = prefix + path(K-1 tokens) + [gt[K-1]]        len = P + K
      start = P
      the engine scores positions P .. P+K-1, and position P+k is predicted from
      seq[:P+k] = prefix + path[:k]. So row k carries p(. | prefix, path[:k]) and
      the quantity we want is row_k[gt[k]].
    The trailing gt[K-1] exists only to create position P+K-1; it is never
    conditioned on by any row we read.
    """
    P = len(prefix)
    seqs = [list(prefix) + p + [gt[K - 1]] for p in paths]
    rows = eng.teacher_forced_query(seqs, [P] * len(seqs), [list(gt)] * len(seqs))
    out = []
    for r in rows:
        out.append([math.exp(r[k][gt[k]]) for k in range(K)])
    return out          # [path][slot]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--corpus", required=True, choices=sorted(C.CORPORA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--m-base", type=int, default=C.MAIN_M)
    ap.add_argument("--m-max", type=int, default=C.M_MAX)
    ap.add_argument("--limit", type=int, default=0, help="smoke/pilot")
    B.add_engine_args(ap)
    BA.add_api_args(ap)
    args = ap.parse_args()

    policy = C.CORPORA[args.corpus]
    K = C.GAMMA

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

    anchors = [json.loads(l) for l in open(args.anchors) if l.strip()]
    if args.limit:
        # the anchor file is written in stratum order, so a raw head slice would
        # run the pilot on one cell; take a stratified slice instead
        by = collections.defaultdict(list)
        for a in anchors:
            by[a["stratum"]].append(a)
        rng = random.Random(C.SEED)
        per = max(1, args.limit // max(1, len(by)))
        picked = []
        for s in sorted(by):
            rng.shuffle(by[s])
            picked += by[s][:per]
        anchors = picked[: args.limit]
        print(f"pilot: stratified slice of {len(anchors)} anchors over "
              f"{len(by)} strata", flush=True)

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)
    print(f"stop token ids: {stop_ids}", flush=True)

    max_ctx = max(seqs[a["prompt_id"]]["prompt_len"] + a["t"] + K
                  for a in anchors) + 8

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_esc = n_ambiguous = n_jk_bad = n_written = 0
    n_skipped = 0

    args.context_length = args.context_length or max_ctx
    with BA.make_target(args, need_logprobs=True) as eng, out.open("w") as fout:
        print(f"engine up; context_length={args.context_length or max_ctx}",
              flush=True)

        for i, a in enumerate(anchors):
            seq = seqs[a["prompt_id"]]
            full = seq["prompt_ids"] + seq["response_ids"]
            cut = seq["prompt_len"] + a["t"]
            if cut + K > len(full):
                n_skipped += 1
                continue
            prefix, gt = full[:cut], full[cut: cut + K]

            # Put the prefix in the radix tree before anything fans out over
            # it; otherwise the M sampling requests are admitted together
            # against a cold tree and each prefills the whole prefix.
            eng.warm_prefix(prefix)

            # ---- CE_A: teacher forced on the TRUE path -------------------
            nll = eng.teacher_forced_nll([prefix + gt], [len(prefix)])[0]
            ce_a = list(nll)                      # already -log p, length K

            # ---- CE_B / CE_commit: M sampled paths -----------------------
            # anchor-specific seed base so the ladder's extra paths are fresh
            # draws rather than repeats of the first M. NOT builtin hash(): it
            # is salted per interpreter for str, so the run would be
            # irreproducible across processes.
            sb = anchor_seed(a["prompt_id"], a["t"], C.SEED)
            paths, nlive = sample_paths(eng, prefix, args.m_base, policy, K,
                                        stop_ids, sb)
            probs = score_paths(eng, prefix, gt, paths, K)
            pg = [[probs[m][k] for m in range(len(probs)) if k <= nlive[m]]
                  for k in range(K)]

            m = args.m_base
            while m < args.m_max and needs_escalation(pg, ce_a):
                more, nl = sample_paths(eng, prefix, m, policy, K, stop_ids,
                                        sb + m * 7919)
                pr = score_paths(eng, prefix, gt, more, K)
                for k in range(K):
                    pg[k] += [pr[mm][k] for mm in range(len(pr)) if k <= nl[mm]]
                nlive += nl
                m *= 2
                n_esc += 1
            ce_b_plug, ce_b, se, jk_bad = [], [], [], []
            for k in range(K):
                p, c, s, ok = ce_b_jackknife(pg[k])
                ce_b_plug.append(p); ce_b.append(c); se.append(s)
                if not ok:
                    jk_bad.append(k)
            n_jk_bad += bool(jk_bad)
            ce_commit = [sum(-math.log(max(x, 1e-12)) for x in pg[k]) / len(pg[k])
                         if pg[k] else None for k in range(K)]

            dce = [b - x for b, x in zip(ce_b, ce_a)]
            # slots where MC noise could still flip the eps_info classification
            # after the ladder ran. This is the diagnostic that replaces
            # `converged`: it counts anchors that actually threaten a published
            # number, rather than anchors that merely have a large SE somewhere.
            amb = ambiguous_slots(dce, se)
            n_ambiguous += bool(amb)

            fout.write(json.dumps({
                **{q: a[q] for q in ("prompt_id", "t", "context", "stratum",
                                     "pi", "N_s", "n_s")},
                "corpus": args.corpus,
                "M": m,
                "ambiguous_slots": amb,
                "jk_unstable_slots": jk_bad,
                "paths_alive_at_end": int(sum(1 for v in nlive if v >= K - 1)),
                "paths_scored_per_slot": [sum(1 for v in nlive if k <= v)
                                          for k in range(K)],
                "ce_A": ce_a,
                "ce_B": ce_b,              # jackknife-corrected: the one to use
                "ce_B_plugin": ce_b_plug,
                "ce_B_se": se,
                "ce_commit": ce_commit,
                "dCE": dce,
            }) + "\n")

            n_written += 1
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(anchors)}  escalations={n_esc}  "
                      f"threshold-ambiguous={n_ambiguous}", flush=True)

    # HARD FAIL on an incomplete run. A crashed arm leaves a short-or-empty
    # file behind, and every downstream reader treats "file exists" as "arm
    # succeeded" -- a half-finished run that still produces output is far more
    # dangerous than one that produces none. Checked here so the exit code is
    # the signal, not a line in a log someone has to notice.
    expected = len(anchors) - n_skipped
    if n_written != expected:
        raise SystemExit(
            f"INCOMPLETE: wrote {n_written} records, expected {expected} "
            f"({len(anchors)} anchors - {n_skipped} legitimately skipped). "
            f"{out} is unusable; delete it and rerun.")

    print(f"== {args.corpus} -> {out}: {len(anchors)} anchors, M={args.m_base} "
          f"base, {n_esc} boundary escalation step(s)", flush=True)
    if n_jk_bad:
        print(f"   {n_jk_bad} anchor(s) had an unusable jackknife correction "
              f"(one path dominating the\n   mean); those slots fall back to the "
              f"plug-in and carry `jk_unstable_slots`.", flush=True)
    print(f"   {n_ambiguous} anchor(s) still have a slot within "
          f"{C.AMBIGUITY_K}*SE of eps_info={C.INFORMATIVE_THRESHOLD}.", flush=True)
    print( "   Those are the only ones whose informative/not classification is "
           "noise-limited;\n   stats.py quotes this count alongside every "
           "incidence figure. A large SE far\n   from the threshold does not "
           "threaten any published number and is not counted.", flush=True)


if __name__ == "__main__":
    main()

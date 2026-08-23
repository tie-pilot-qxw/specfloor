"""Eligible-anchor population, stratification, and weighted sampling.

The population is built from COMPLETED trajectories. Anchors are never chosen
while generating -- that is what makes the inclusion probabilities, and
therefore the workload-native aggregate, well defined.

Two coordinates per anchor, and they are not the same variable:
  absolute prefix length  c = prompt_len + t        (how much context the
                                                     target has processed)
  relative response position u = t / response_len   (where in the answer)

Every selected anchor stores pi = n_s / N_s so aggregates can be recovered by
inverse-probability weighting.

Usage:
  python -m specfloor.anchors --corpus-file runs/C0/gsm8k.jsonl \
      --out runs/C0/gsm8k.anchors.jsonl --budget 2500
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

from specfloor import config as C

# Special/control tokens come from the TOKENIZER, never from a vocabulary range.
# markov_order_probe.py:158 documents why they matter: a handful of anchors whose
# block straddles a control token carried ~40% of the PB tail mass. A block
# containing one is not a normal speculative block.
#
# This used to be `id >= 151643`, which is where Qwen3's control block starts.
# On Qwen that is right to the token -- its 14 special ids are 151643..151656 and
# the rest of the band is reserved and never generated (0 occurrences in 765,962
# corpus tokens). On any other family it is nonsense in both directions: Gemma-4
# has a 262,144-token vocabulary whose control tokens sit at ids 0..52 and whose
# ordinary text runs far past 151643, so the range test dropped 75%-85% of every
# candidate block while missing every control token it was meant to catch.


def special_ids(target: str) -> frozenset:
    """The tokenizer's own control tokens.

    `all_special_ids` covers the named roles (bos/eos/pad/unk and the extras);
    `added_tokens_decoder` covers everything registered as an added token, whose
    `.special` flag is what chat templates key off. Neither alone is complete on
    both families, so take the union. Refuse to proceed if it is empty rather
    than fall back to a guess -- an empty set silently disables the filter.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    ids = {int(i) for i in (tok.all_special_ids or []) if i is not None}
    for i, t in (getattr(tok, "added_tokens_decoder", None) or {}).items():
        if getattr(t, "special", False):
            ids.add(int(i))
    if not ids:
        raise SystemExit(
            f"{target}: the tokenizer exposes no special tokens. Refusing to "
            f"guess a vocabulary range -- see the note above anchors.special_ids.")
    return frozenset(ids)


def think_close_id(target: str):
    """The id that closes a reasoning trace, or None if the family has none."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(target)
    i = tok.convert_tokens_to_ids("</think>")
    unk = getattr(tok, "unk_token_id", None)
    return None if i is None or i == unk or i < 0 else int(i)


def context_bucket(c: int) -> str:
    for lo, hi in C.CONTEXT_BUCKETS:
        if lo <= c < hi:
            return f"[{lo},{hi})"
    return "overflow"


def relative_bin(u: float) -> str:
    for name, lo, hi in C.RELATIVE_BINS:
        if lo <= u < hi:
            return name
    return C.RELATIVE_BINS[-1][0]


def eligible(seq: dict, special: frozenset, close_id=None,
             gamma: int = C.GAMMA):
    """Yield every position that admits a complete, special-token-free block.

    A right-censored trajectory has no true EOS, so its trailing block is not a
    real block -- drop the last gamma positions rather than the last one.

    Blocks containing a special/control token are dropped outright: they are not
    speculative blocks the serving path would ever verify, and historically a
    few of them dominated the tail.

    For thinking-on corpora the reasoning trace and the answer are different
    regimes, so `in_think` is recorded and `u` is measured within whichever
    segment the anchor falls in -- otherwise `early/middle/late` would silently
    mean "position in the reasoning trace" for C2 and "position in the answer"
    for C0/C1 while printing under the same stratum keys.
    """
    ids = seq["response_ids"]
    L = seq["response_len"]
    usable = L - gamma - (gamma if seq.get("right_censored") else 0)
    close = (ids.index(close_id)
             if close_id is not None and close_id in ids else None)

    n_special = 0
    for t in range(usable + 1):
        block = ids[t: t + gamma]
        if block and not special.isdisjoint(block):
            n_special += 1
            continue
        c = seq["prompt_len"] + t
        if close is None:
            in_think, u = False, t / max(1, L)
        elif t <= close:
            in_think, u = True, t / max(1, close)
        else:
            in_think, u = False, (t - close) / max(1, L - close)
        yield {
            "prompt_id": seq["prompt_id"],
            "t": t,
            "context": c,
            "in_think": in_think,
            "cell": (context_bucket(c), relative_bin(u)),
        }
    if n_special:
        seq["_dropped_special"] = n_special


def build_population(path: str, special: frozenset, close_id=None):
    pop, dropped, n_seq = [], 0, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = json.loads(line)
            n_seq += 1
            pop.extend(eligible(seq, special, close_id))
            dropped += seq.get("_dropped_special", 0)
    if dropped:
        print(f"  dropped {dropped:,} anchors whose block contained one of "
              f"the tokenizer's {len(special)} special tokens across {n_seq} "
              f"sequences "
              f"({dropped/(dropped+len(pop))*100:.2f}% of candidates)")
    return pop


def sample(pop, budget: int, seed: int = C.SEED,
           estimand: str = C.ESTIMAND):
    """Stratified sample with EXACT inclusion probabilities.

    block-weighted (primary): simple random sampling without replacement inside
      each stratum, so pi = n_s / N_s is exactly right and 1/pi recovers the
      block-weighted (workload-native) estimand.

      The earlier design shuffled distinct SEQUENCES, kept n_s of them, then took
      one uniform position from each. That is a two-stage scheme whose true
      inclusion probability is (n_s/S_cell)*(1/m_j) -- it varies by sequence
      length, so recording n_s/N_s silently computed a within-cell SEQUENCE
      average instead. Per-sequence caps and non-overlap rejection are what made
      pi intractable, so they are gone: within-sequence dependence is a variance
      problem, and the prompt-level hierarchical bootstrap already covers it.

    sequence-weighted (robustness): one uniform position per sequence, each
      sequence carrying equal weight, so every anchor gets weight 1 (pi = 1.0).
      Recording 1/m_j here would have made 1/pi = m_j and reproduced the
      block-weighted estimand -- i.e. the two arms would have agreed by
      construction, which is not robustness.
    """
    rng = random.Random(seed)

    if estimand == "sequence":
        by_seq = collections.defaultdict(list)
        for a in pop:
            by_seq[a["prompt_id"]].append(a)
        keys = sorted(by_seq)
        rng.shuffle(keys)
        return [dict(rng.choice(by_seq[k]), pi=1.0, stratum="sequence",
                     N_s=len(by_seq[k]), n_s=1)
                for k in keys[:budget]]

    by_cell = collections.defaultdict(list)
    for a in pop:
        by_cell[a["cell"]].append(a)

    populated = sorted(by_cell)
    if not populated:
        return []
    per_cell = max(1, budget // len(populated))

    chosen, short = [], []
    for cell in populated:
        bucket = by_cell[cell]
        N_s = len(bucket)
        n_s = min(per_cell, N_s)
        if n_s < per_cell:
            short.append((cell, n_s, per_cell))
        for a in rng.sample(bucket, n_s):        # SRSWOR -> pi is exact
            chosen.append(dict(a, pi=n_s / N_s, stratum=f"{cell[0]}|{cell[1]}",
                               N_s=N_s, n_s=n_s))
    if short:
        print(f"\n  BUDGET SHORTFALL: {len(short)} cell(s) had fewer eligible "
              f"anchors than the per-cell target ({per_cell}):")
        for cell, got, want in short:
            print(f"    {cell[0]:<16} {cell[1]:<7} {got}/{want}")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=C.CHEAP_ANCHORS_PER_DOMAIN)
    ap.add_argument("--estimand", default=C.ESTIMAND, choices=("block", "sequence"))
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--target", required=True,
                    help="the model whose tokenizer defines the special tokens; "
                         "a vocabulary range is not portable across families")
    args = ap.parse_args()

    special = special_ids(args.target)
    close_id = think_close_id(args.target)
    print(f"special tokens from {args.target}: {len(special)} ids, "
          f"{min(special)}..{max(special)}"
          + (f"; </think> = {close_id}" if close_id is not None else ""))
    pop = build_population(args.corpus_file, special, close_id)
    print(f"eligible anchor population: {len(pop):,}")
    hist = collections.Counter(a["cell"] for a in pop)
    for cell in sorted(hist):
        print(f"  {cell[0]:<16} {cell[1]:<7} {hist[cell]:>9,}  "
              f"({hist[cell]/len(pop)*100:5.2f}% of population)")

    chosen = sample(pop, args.budget, args.seed, args.estimand)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for a in chosen:
            f.write(json.dumps(a) + "\n")

    print(f"\nsampled {len(chosen):,} anchors ({args.estimand}-weighted)"
          f"  [requested {args.budget:,}]")
    if len(chosen) < 0.9 * args.budget:
        print(f"  !! realised sample is {len(chosen)/args.budget:.0%} of the "
              f"requested budget -- thin cells, see the shortfall list above")
    sel = collections.Counter(a["stratum"] for a in chosen)
    for s in sorted(sel):
        print(f"  {s:<26} {sel[s]:>6}")
    thin = [s for s in sel if sel[s] < C.MIN_INFORMATIVE_PER_CELL]
    if thin:
        print(f"\n  cells below the headline minimum "
              f"({C.MIN_INFORMATIVE_PER_CELL}) -- report incidence and mean "
              f"only, mark exploratory:")
        for s in thin:
            print(f"    {s} (n={sel[s]})")


if __name__ == "__main__":
    main()

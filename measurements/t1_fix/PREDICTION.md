# Written before the run, so it can be wrong

`probe_tk` built its m>=1 sequences as

    prefix + path[:k-m] + gt[k-m:k]

and read `tops[-1]` as slot `k`. The backend returns one row per token and row
`i` is the distribution that PREDICTED token `i`, so `tops[-1]` is the row for
the last REVEALED token: `p(. | X, s)`, computed from a context that does not
contain `z*`. The importance weight `w(s) = p(z* | X, s)` is read from that same
row and is correct. So what the run reported at m=1 was

    min_q  sum_s w(s) TV( p(. | X, s), q ) / sum_s w(s)

-- the weighted spread of the slot-(k-1) predictive distributions, each weighted
by its own mass at `z*` -- and not

    min_q  sum_s w(s) TV( p(. | X, s, z*), q ) / sum_s w(s) ,

which is `T_k^{(1)}`. Off by one slot. The fix appends `gt[k]` so the row at
position `len(prefix)+k` exists; `starts` is unchanged, so `v[:m]` is still the
weight and the trailing token's identity never enters an estimate.
`probe_rm.ce_mixed_all` has always built the sequence this way.

Nothing at m=0 is affected: rung 0 goes through `teacher_forced_topk` on the
sampled path itself, where row `i` predicting token `i` IS slot `i`. `probe_rpre`
and `probe_api_floor` are unaffected by construction -- they read `p` from the
forward that samples the token, so there is no position to align.

## The prediction

The broken quantity weights each `p(.|X,s)` by its own mass at `z*`, which
upweights exactly the prefixes that already agree at that coordinate. That is a
concentrating weight, so the broken number should be **too small**, and it was:
0.0185/0.0210 at slot 6 against grouping's 0.0327/0.0413.

**If the off-by-one is the whole story**, fixed SNIS at slot 6 lands in
grouping's range -- roughly **0.030-0.045** -- the paired difference against
grouping loses significance, and SNIS becomes **monotone in the slot index**,
which it currently is not (it falls 0.0217 -> 0.0185 from slot 5 to 6).

**If fixed SNIS still reads about half of grouping**, the off-by-one was real but
not the explanation, and the two estimators genuinely disagree for a reason not
yet identified.

**If fixed SNIS overshoots grouping**, the paper's choice to quote the larger of
the two stops being the conservative one and Q3 has to be restated.

The slot-1 identity `T_1^{(1)} = 0` must still hold exactly: at k=m=1 the free
segment is empty, so all sequences coincide whatever token is appended.

Either of the last two is reportable. What is not acceptable is keeping the
current 0.0185/0.0210 anywhere in the paper, or keeping the midpoint-drift
argument built on it.

## What is run

    probe_tk --rungs 0,1 --split --top-k 256, target Qwen3-4B, four domains,
    the same 96 anchors per domain as t1/, at --paths 1024 and --paths 256

Same corpora, same ladder, same anchors, same seeds as `t1/`. The only change is
the trailing token in the sequence, so the comparison against `t1/` is paired
per (anchor, slot). Run under `specfloor` rather than `DeepSpec/measurement`;
the two copies of probe_tk differ only in imports and docstrings.

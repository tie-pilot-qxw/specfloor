# Written before the run, so it can be wrong

`anchors.py` dropped any block containing a token id `>= 151643`. That is where
Qwen3's control block starts; on Qwen it is exact, and the 279 reserved ids above
its 14 real special tokens occur **0 times** in 765,962 corpus tokens, so the
tokenizer-derived filter reproduces the Qwen eligible population **exactly**
(319,304 both ways, symmetric difference 0). No Qwen result moves.

Gemma-4 has a 262,144-token vocabulary. Ordinary text runs far past 151643 and
its control tokens sit at ids 0..52, so the range test dropped 75%-85% of every
candidate block *and* missed every control token it existed to catch:

    eligible population   old 42,614    correct 249,915    kept 17.1%

So the published Gemma row is measured on a sixth of its population, selected by
token id -- which on a BPE vocabulary is a proxy for token frequency, hence for
how ordinary the text is. Rarer tokens have higher ids, so the retained blocks
are the ones built from common tokens.

## The prediction

Common tokens are the predictable ones, so the retained sixth should have been
**easier** than the population, and the corrected floor and risk should come out
**higher**:

    published:  T^(0)_6 = 0.3322   R_6 = 0.7880   G/R = 57.8%
    predicted:  T^(0)_6 rises, R_6 rises, and G/R moves by less than either,
                since both sides move together.

**If G/R moves by more than about 5 points**, the Gemma row's contribution to
Sec 6's "43%-58%" range changes and the range must be restated.

**If T^(0)_6 rises above R_6's published value**, or if the corrected numbers
land outside the other three targets' spread, the cross-family claim needs
rewriting rather than renumbering.

**If nothing moves materially**, that is worth stating too: it would mean the
id-based selection was close to ignorable for these quantities, and the row
stands as published with a corrected provenance.

## What is run

Corpus REUSED from `scale/gemma12b/C0` -- the bug is in anchor selection inside
an already-written response, so the responses are unaffected. Rerun from
`anchors` through `probe_cheap`, `probe_tk` and both `probe_rpre` orders, since
each consumes the anchor set. `probe_tk`'s m>=1 output carries the separate
slot-alignment fix; only its order-0 column feeds the scale table.

"""specfloor -- information floors and model gaps for block speculative decoding.

A block drafter emits every slot of a block from one forward pass, so each slot
must commit to a distribution before the target's realisations at earlier slots
exist. This package measures what that costs, and separates it from what the
drafter costs.

The whole thing rests on one identity. Under the speculative accept rule the
probability that a drafted token survives verification is

    alpha(p, q) = sum_v min(p(v), q(v)) = 1 - TV(p, q),

so rejection loss is a total variation distance, and one may ask how small it
could *possibly* be for a proposal restricted to see only the last m realised
tokens. That minimum is the **information floor** T^(m). It contains no drafter:
it is a property of the target and the factorisation, and every drafter of that
shape must pay it. What a real drafter loses above its own floor is the **model
gap** G = R - T^(m).

Two halves, two dependency sets
-------------------------------
Measuring a floor needs the **target alone** -- sample continuations, form the
realisation family, solve the minimisation -- so it needs nothing beyond
`transformers` and an inference engine, or in the API case not even that.
Measuring a gap additionally needs a real drafter's proposal, which means
loading and running a DFlash/DSpark checkpoint, and that code lives in DeepSpec.

Everything here therefore imports cleanly without DeepSpec. Only `probe_rpre`,
`probe_br` and `eval_nll` reach for it, lazily, through `specfloor._deepspec`,
and a missing install fails with an instruction rather than a traceback.

Two things this package will not do
-----------------------------------
It will not silently degrade. A probe that needs a drafter and cannot find one
raises; a cell whose importance weights are too degenerate to trust is dropped
and counted rather than reported; a truncated read that cannot resolve a cell
does not guess it. Every gate reports the population it removed.

And it never mixes the free-rollout law with the survival-conditioned one. R and
T^(m) average over the target's own paths, while accepted length weights a slot
by the probability of surviving to it. Those are different populations,
`srv_report` measures how far apart they are, and nothing here quietly converts
between them.
"""

__version__ = "0.1.0"

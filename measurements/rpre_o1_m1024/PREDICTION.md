# Written before the run, so it can be wrong

`T^(1)` at slot 6 on Qwen3-4B is estimated two ways and they differ by 2.2x:

    (A) SNIS, M=1024      fit-and-score 0.0185   held-out 0.0210
    (B) grouping, M=256   fit-and-score 0.0327   held-out 0.0413

    paired difference (B held-out - A) = +0.0228, 95% CI [+0.0109, +0.0387]

The marginal bootstrap intervals nearly overlap, so this is not visible without
pairing; the paired test says it is real. §6.4a attributes most of it to how
many paths land in each conditioning group, which is the sample (B) has to
solve each inner minimisation from. Binned by that occupancy at slot 6:

    mean paths/group   cells   fit-and-score   held-out   bracket
        6-9              19       0.1082        0.1478    0.0396
        9-14             30       0.0895        0.1207    0.0312
        14-25            63       0.0434        0.0544    0.0110
        >=25            267       0.0204        0.0228    0.0024

117 of 384 cells sit below 25 paths per group, and that is where the two
estimators part. Occupancy is proportional to M, so this run at M = 1024
multiplies it by four: a cell at 6-9 moves to 24-36, into the bin where the
bracket is 0.0024 and (B) reads what (A) reads.

## The prediction

**If the disagreement is resolution**, (B) at M = 1024 falls from 0.0413 to
roughly **0.022-0.026** held-out, its own fit-and-score-to-held-out bracket
narrows from 0.0086 to under 0.003, and the paired difference against (A)
loses its sign or shrinks below 0.005.

**If (B) stays near 0.04**, occupancy was not the cause. Then the suspect is
(A): forcing one revealed predecessor per anchor and reweighting 1024 early
paths onto it thins the effective sample in a way its own held-out column does
not catch, and the paper should quote (B) and say so.

Either outcome is reportable. What is not acceptable is quoting 0.041 as the
floor without knowing which of these is true, which is what §5 did before.

## What is run

    probe_rpre --order 1 --cond both --paths 1024 --split
        target Qwen3-4B, drafter dspark_qwen3_4b_block7,
        four domains, the same 384 anchors as rpre_o1/, full vocabulary

Same anchors, same corpus files, same seeds. Only `--paths` changes, so the
comparison against `rpre_o1/` is paired per (anchor, slot).

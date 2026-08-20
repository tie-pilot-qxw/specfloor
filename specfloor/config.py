"""Frozen measurement constants. See paper/PROTOCOL.md v2.

Nothing in this file may change once the main run starts. Every probe imports
from here rather than taking its own default, so a config drift is impossible
by construction.
"""

from __future__ import annotations

# ---------------------------------------------------------------- contract --
GAMMA = 7                       # block size; the released checkpoint's gamma
TARGET = "Qwen/Qwen3-4B"

DOMAINS = ("gsm8k", "mbpp", "alpaca", "arena-hard-v2")
REASONING_DOMAINS = ("gsm8k",)  # C2 seed; a harder workload is appended if needed

PROMPTS_PER_DOMAIN = 1024

# ------------------------------------------------------------ generation ----
# Per-corpus policy. The whole point of v2 is that these differ; never merge.
CORPORA = {
    # T=1 untruncated. Two things at once, which is why it is the primary arm:
    # CE really is a conditional entropy estimate here, AND it is the drafter's
    # own evaluation protocol -- eval.py defaults to --temperature 1.0 and
    # neither eval.py nor deepspec/eval/ mentions top_p or top_k anywhere, so
    # deepspec/utils/sampling.py samples temperature-scaled softmax untruncated.
    "C0": dict(thinking=False, temp=1.0, top_p=1.0, top_k=0),
    # TRAINING-matched, not serving-matched. 0.7 is what
    # scripts/data/generate_train_data.py generates the drafter's training data
    # at; (0.8, 20) is Qwen3's recommended non-thinking sampling. Kept as a
    # robustness arm: it answers whether a result survives a truncated law, not
    # whether it holds at deployment -- C0 answers that.
    "C1": dict(thinking=False, temp=0.7, top_p=0.8, top_k=20),
    # long-thinking: PART I QUANTITIES ONLY (drafter is OOD here)
    "C2": dict(thinking=True, temp=1.0, top_p=1.0, top_k=0),
}
PART_I_ONLY = ("C2",)

SAFETY_MAX_NEW_TOKENS = 16384   # runtime valve, NOT the experiment definition
MAX_CENSOR_RATE = 0.005         # raise the valve if exceeded

# ---------------------------------------------------------------- strata ----
CONTEXT_BUCKETS = ((0, 512), (512, 2048), (2048, 4096),
                   (4096, 8192), (8192, 16384), (16384, 1 << 30))
RELATIVE_BINS = (("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01))

ESTIMAND = "block"              # primary; "sequence" is the robustness arm
# NOTE: there is deliberately NO per-sequence cap and no non-overlap rule.
# Any such constraint makes the inclusion probability intractable and turns the
# block-weighted estimand into a sequence-weighted one. Within-sequence
# dependence is a variance problem and the prompt-level bootstrap covers it.

# --------------------------------------------------------------- budgets ----
CHEAP_ANCHORS_PER_DOMAIN = 2500     # dCE / incidence / commitment penalty
RM_ANCHORS_PER_DOMAIN = 512         # R_m: prefill-bound, so subsample instead
MIN_INFORMATIVE_PER_CELL = 128      # below this: no conditional quantiles
PREFERRED_INFORMATIVE_PER_CELL = 200

# ------------------------------------------------- eligibility thresholds ---
# TWO thresholds, for two different estimands. They are deliberately different
# and neither is a relaxation of the other.
INFORMATIVE_THRESHOLD = 0.01        # eps_info: incidence, tails, conditional dCE
RM_ELIGIBILITY_THRESHOLD = 0.05     # eps_R: R_m only
# R_m = (CE_B - CE_M) / (CE_B - CE_A) is a NORMALISED ratio, so a near-zero
# denominator makes it meaningless -- at dCE = 0.015 with 0.01 nats of estimation
# error the ratio wanders anywhere from 0 to >1. Anchors that qualify as
# informative but cannot support a stable ratio are therefore excluded from R_m
# and ONLY from R_m. Running the expensive pass on them was pure waste: in the
# pilot 175 of 324 R slots came back None for exactly this reason.

# ---------------------------------------------------------- monte carlo -----
# NO per-anchor SE target. It was the wrong stopping rule: the required M is
# extremely heavy-tailed (pilot: median 0, p75 129, p90 855, max 19537 to reach
# SE < 0.05), so chasing it burns the budget on anchors that no feasible M would
# fix. The paper's claims are population-level -- E[dCE], P(dCE > eps), the tail
# shape -- not per-anchor accuracy.
# MAIN_M = 512, SET BY THE LADDER -- not by intuition. The first draft guessed
# 64; the ladder rejected 64, then 128, then 256.
#
# Criterion (PROTOCOL.md sec.4): MC approximation error as a fraction of the
# SAMPLING CI half-width at the richer arm. <=0.25 passes (total uncertainty
# inflates by sqrt(1+0.25^2) = 3%), >=0.5 fails. NOT statistical significance of
# the difference: the arms are nested, so the paired CI is tight enough that any
# real shift excludes zero and that test would reject every affordable M.
#
# Converged M per domain, each verified against the NEXT rung up -- the top rung
# of a ladder can never certify itself:
#     gsm8k    128   (vs 512)
#     alpaca   256   (vs 512)
#     arena    256   (vs 512, on the 8k corpus; see the censoring note below)
#     mbpp     512   (vs 1024)
#
# THE REQUIREMENT IS STATISTIC-DEPENDENT, and this is the useful finding.
# On mbpp, measured against M=1024:
#     P(dCE>eps)     0.12 at M=128     already converged
#     p50 | inf      0.07 at M=128     already converged
#     p90 | inf      0.17 at M=128     already converged
#     E[dCE]         0.67 at M=128, 0.76 at M=256   needs 512
#     E[dCE | inf]   0.65 at M=128, 0.92 at M=256   needs 512
# Threshold and quantile statistics converge at 128 in EVERY domain; only the
# means need 512. That is structural, not incidental: the mean of a heavy-tailed
# variable is dominated by the extreme tail, which is exactly what Monte Carlo
# estimates worst, while quantiles are tail-robust by construction. If the paper
# headlines incidence and p90, M=128 would suffice; 512 is the price of quoting
# E[dCE] with the same rigour.
#
# 512 is the global maximum across domains, taken deliberately rather than
# per-domain: a uniform M keeps cross-domain comparisons free of one more
# implementation difference. Cost is 4x M=128 on the cheap pass (~7 min per 252
# anchors), i.e. an overnight run rather than an afternoon one.
MAIN_M = 512
M_BASE = MAIN_M
# Must exceed MAIN_M or boundary escalation can never fire -- with M_MAX == MAIN_M
# the `while m < m_max` loop is dead on arrival and the one adaptive rule the
# protocol keeps would silently do nothing.
M_MAX = 1024
M_LADDER = (32, 64, 128, 256, 512, 1024)   # calibration only, never the main run
CALIBRATION_ANCHORS = 256
# At M >= 256 an unbounded fan-out killed the engine (SIGQUIT, child died). Pass
# --max-running to bound in-flight requests; 64 was sufficient.
MAX_RUNNING_AT_HIGH_M = 64

# The ONE adaptive rule that survives, and why. Escalating near the
# classification boundary is what keeps MC noise and the eps_info threshold in
# disjoint regions -- the property that makes a thresholded statistic safe
# despite noisy individual anchors. In the pilot it fired on 3 anchors, versus
# 12 for the blanket SE floor: nearly free, and it is the mechanism, not the
# cost. stats.py reports the residual ambiguity so this is checked, not assumed.
BOUNDARY_K = 2.0                    # escalate while |dCE - eps_info| < K * SE
AMBIGUITY_K = 2.0                   # report anchors with |dCE - eps| < K * SE

# A PERMANENT INVARIANT, not a one-time check. With M_MAX == MAIN_M the
# `while m < m_max` escalation loop is dead on arrival: the protocol still
# documents boundary escalation, the config still lists BOUNDARY_K, and nothing
# ever fires. That is the textbook shape of protocol/code drift, and it is
# silent -- the run completes and looks normal.
assert M_MAX > MAIN_M, (
    f"M_MAX ({M_MAX}) must exceed MAIN_M ({MAIN_M}), or boundary escalation "
    f"can never fire and the one adaptive rule in the protocol is a no-op.")

RM_ORDERS = (1, 2, 4)
RM_MIXED_PATHS = 24
# Sample splitting: the R_m pass re-draws its own paths rather than reusing the
# cheap pass's CE_B. Selecting on an estimate and then putting that SAME estimate
# in the ratio is selection-on-noise, and because CE_B appears in BOTH numerator
# and denominator the induced bias drives R_m toward 1 -- i.e. toward the H2
# conclusion. The offset makes the redraw independent of the selection.
RM_RESCORE_SEED_OFFSET = 0x5F3759DF

# ---------------------------------------------------------------------------
# Below this line: constants for stages not yet implemented in this package.
# They are NOT enforced by any code today -- do not assume otherwise.
#   DOMAINS / REASONING_DOMAINS   run-order bookkeeping (driver scripts)
#   CALIBRATION_*, M_LADDER       the MC convergence pilot (README step 3)
#   PAIRED_*                      the C2 long-context paired cohorts
#   EQUIVALENCE_DELTA             set only after the harness noise calibration
#   PREFERRED_*                   advisory targets, the MIN_ variants are enforced
#
# The M ladder is now the PRIMARY convergence evidence rather than a
# nice-to-have: with no per-anchor SE gate, the justification for MAIN_M is that
# the AGGREGATE statistics (mean dCE, incidence, p50/p90, domain ranking) do not
# move across M = 32/64/128/256 on the calibration subset. Run it before the
# main pass and record the numbers.
# ---------------------------------------------------------------------------

# ------------------------------------------------------------- inference ----
BOOTSTRAP_B = 10_000
BOOTSTRAP_OUTER = "prompt"          # documentation; stats.py hard-codes prompt
CI = 0.95

# -------------------------------------------------------- long context ------
PAIRED_LADDER = ((512, 2048), (2048, 4096), (4096, 8192))
MIN_PAIRED_SEQUENCES = 128
PREFERRED_PAIRED_SEQUENCES = 256

# ------------------------------------------------- capacity equivalence -----
# delta is NOT fixed here: it must be calibrated against the harness's own
# reproducibility first (same checkpoint, two seeds). See PROTOCOL.md sec.7.
EQUIVALENCE_DELTA = None            # set after calibration, then frozen

SEED = 20260818

r"""Is $T_1^{(1)}$ zero, and if not, what is the non-zero made of?

$Z_{<1} = Z_0$, so conditioning on the realised token at slot 0 fixes the ENTIRE
realisation that slot 1 depends on. Inside a group the family $\{p_Z\}$ is a
singleton and $T_1^{(1)} = 0$ identically, for every anchor and every $M$. Any
non-zero is therefore an artefact, and this script measures which one.

Two candidates, distinguished by a single number.

  INDEXING. If the conditioning vector were misaligned -- off by a slot, or
  carrying rows from a different anchor -- rows inside a "group" would be
  genuinely different distributions and their spread would be of the same order
  as the spread ACROSS groups.

  ARITHMETIC. A batched bf16 decode computes the same logits row at different
  positions of the batch, and tile/split-k assignment differs by position, so
  two paths that share $Z_0$ get answers that agree only to bf16. Then the
  within-group spread sits far below the across-group spread and scales like the
  dtype, not like the data.

The discriminator printed below is `within / across`. Near 1 means indexing;
orders of magnitude below 1 means arithmetic. Slot 0 is carried as a control: it
is served from one broadcast row, so its spread is exactly 0 by construction and
anything else would indicate the harness itself is unsound.
"""

from __future__ import annotations

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specfloor import config as C
from specfloor.corpus import resolve_stops
from specfloor.probe_cheap import anchor_seed
from specfloor.probe_rpre import barycentre, mean_tv, rollout


def spread(P):
    """Mean pairwise L1 between rows -- the quantity that must vanish in a group."""
    n = P.shape[0]
    if n < 2:
        return None
    tot = cnt = 0.0
    for i in range(n - 1):
        tot += (P[i + 1:] - P[i].unsqueeze(0)).abs().sum(dim=1).sum().item()
        cnt += n - 1 - i
    return tot / cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--anchors", type=int, default=6)
    ap.add_argument("--paths", type=int, default=256)
    ap.add_argument("--kv-budget-gib", type=float, default=4.0)
    args = ap.parse_args()

    device = "cuda"
    policy = C.CORPORA["C0"]
    K = C.GAMMA
    seqs = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(args.corpus_file) if l.strip()}
    anchors = [json.loads(l) for l in open(args.cheap) if json.loads(l)["prompt_id"] in seqs]
    rng = random.Random(C.SEED)
    rng.shuffle(anchors)
    anchors = anchors[: args.anchors]

    tok = AutoTokenizer.from_pretrained(args.target)
    stops = resolve_stops(args.target, tok)
    tgt = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    print(f"{'anchor':>6} {'chunk':>6} | {'slot0 spread':>13} | {'s1 within':>10} "
          f"{'s1 across':>10} {'ratio':>8} {'groups':>7} | {'T1_plug':>9} {'T0_s1':>8}")
    for a in anchors:
        s = seqs[a["prompt_id"]]
        full = s["prompt_ids"] + s["response_ids"]
        cut = s["prompt_len"] + a["t"]
        if len(full) < cut + K:
            continue
        prefix = torch.tensor([full[:cut]], device=device, dtype=torch.long)
        gen = torch.Generator(device=device)
        gen.manual_seed(anchor_seed(a["prompt_id"], a["t"], C.SEED) % (2 ** 63 - 1))
        # rollout returns (parts, conds, reals, idxs, chunk). This unpacked
        # three and raised ValueError on every invocation, so the one
        # independent check on slot indexing and batched arithmetic had never
        # actually run. It runs now, and CI should keep it running.
        slots, conds, _reals, _idxs, chunk = rollout(
            tgt, prefix, args.paths, K, policy, stops, gen,
            int(args.kv_budget_gib * (1 << 30)))

        P0 = slots[0]
        P1, z0 = slots[1], conds[1]
        vals, inv = torch.unique(z0, return_inverse=True)
        wi, wn, mus = 0.0, 0, []
        for g in range(vals.numel()):
            idx = (inv == g).nonzero(as_tuple=True)[0]
            if idx.numel() >= 2:
                sp = spread(P1[idx])
                wi += sp * idx.numel(); wn += idx.numel()
            mus.append(P1[idx].mean(dim=0))
        within = wi / wn if wn else float("nan")
        M = torch.stack(mus)
        across = spread(M) if M.shape[0] >= 2 else float("nan")
        # T1 plug-in, computed exactly as probe_rpre does it
        tot = 0.0
        for g in range(vals.numel()):
            idx = (inv == g).nonzero(as_tuple=True)[0]
            tot += idx.numel() * mean_tv(P1[idx], barycentre(P1[idx]))
        print(f"{a['t']:>6} {chunk:>6} | {spread(P0):>13.3e} | {within:>10.3e} "
              f"{across:>10.3e} {within/across if across else float('nan'):>8.1e} "
              f"{vals.numel():>7} | {tot/P1.shape[0]:>9.2e} "
              f"{mean_tv(P1, barycentre(P1)):>8.4f}")
        del slots, conds
        torch.cuda.empty_cache()

    print("\nslot-0 spread is 0 by construction (one prefill row broadcast to the batch),")
    print("so it validates the harness rather than the model. The number that decides")
    print("the question is `ratio` = within-group / across-group L1 spread at slot 1.")


if __name__ == "__main__":
    main()

"""Clean drafter evaluation NLL:  L_D = -log q_D(y_realized).

NO decay weighting, NO auxiliary hidden loss, NO position weighting, NO ce/l1
mixture. Deliberately NOT the training objective.

--------------------------------------------------------------------------
WHICH L_D?  Three variants are emitted, because they are different estimands
and the decomposition needs a specific one.

  L_D_base      backbone only; slot0 = ANCHOR token, slots 1..K-1 = MASK.
                This is the drafter's round-0 PATH-MARGINAL predictive -- the
                same information set CE_B assumes. *** This is the one the
                information/model decomposition must use. ***

  L_D_mk0       + markov head applied with the only previous token that is
                genuinely known at serve time (the anchor, feeding slot 0).
                Serving-faithful; slots >0 keep the base logits.

  L_D_mk_tf     + markov head teacher-forced on the TRUE previous tokens.
                An optimistic bound: it hands the drafter the in-block path it
                does not have at serve time. Report as a ceiling, never in the
                decomposition -- doing so would make L_D < CE_B possible for
                reasons that have nothing to do with model quality.

Omitting the markov head altogether would measure an ablation of the shipped
drafter (the released block-7 checkpoint carries ~78M trained markov params);
teacher-forcing it would leak the path. Hence all three, explicitly labelled.
--------------------------------------------------------------------------

Anchor convention, verified against training and serving:
  training  common.py:339   anchor_tokens = gather(input_ids, 1, anchor_positions)
  serving   evaluator.py:118 draft_input_ids[:, 0] = output_ids[:, start]
  labels    modeling.py:555  label_offsets = arange(1, block_size+1)
so with the block scored at full[cut:cut+K], the anchor position is cut-1 and
slot 0 must hold full[cut-1] = prefix[-1].

Context masking: training uses `kv_idx < anchor_pos` (common.py:146) -- STRICTLY
less than, so the target hidden state AT the anchor is never a context key. Its
lm_head projection is p(full[cut] | prefix), i.e. an oracle for slot 0. We build
the real dspark block mask rather than passing None.

Usage:
  python -m specfloor.eval_nll --corpus-file runs/C1/gsm8k.jsonl \
      --anchors runs/C1/gsm8k.anchors.jsonl --corpus C1 \
      --draft deepseek-ai/dspark_qwen3_4b_block7 --out runs/C1/gsm8k.nll.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM

from specfloor import _deepspec

from specfloor import config as C


def check_checkpoint(draft, mask_id: int, tlids: list[int], K: int) -> None:
    """Fail loudly on a mismatched checkpoint instead of measuring nonsense."""
    cfg = draft.config
    got_block = int(getattr(cfg, "block_size", -1))
    got_mask = int(getattr(cfg, "mask_token_id", -1))
    got_tl = list(getattr(cfg, "target_layer_ids", []))
    assert got_block == K, f"checkpoint block_size={got_block} but GAMMA={K}"
    assert got_mask == mask_id, f"checkpoint mask_token_id={got_mask} != {mask_id}"
    assert got_tl == tlids, f"checkpoint target_layer_ids={got_tl} != {tlids}"
    n_layers = int(getattr(cfg, "num_target_layers", 0) or 0)
    if n_layers:
        assert max(got_tl) < n_layers - 1, (
            f"target_layer_ids includes the final target layer ({max(got_tl)} of "
            f"{n_layers}) -- that leaks the next-token prediction"
        )


@torch.no_grad()
def block_nll(draft, target_hidden, prefix_len, anchor_tok, gt, mask_id, K):
    """Returns (base, markov_slot0, markov_teacher_forced) NLL lists."""
    dev = target_hidden.device

    din = torch.full((1, 1, K), mask_id, dtype=torch.long, device=dev)
    din[:, :, 0] = anchor_tok                      # <-- the ANCHOR, not gt[0]

    anchor_pos = torch.tensor([[prefix_len - 1]], device=dev)
    pos = torch.cat([
        torch.arange(prefix_len, device=dev).unsqueeze(0),
        _deepspec.create_position_ids(anchor_pos, K),
    ], dim=1)

    block_mask = _deepspec.create_dspark_attention_mask(
        anchor_positions=anchor_pos,
        block_keep_mask=torch.ones(1, 1, dtype=torch.bool, device=dev),
        seq_len=prefix_len,
        block_size=K,
        device=dev,
    )

    emb = draft.embed_tokens(din.reshape(1, K))
    h = draft._forward_backbone(
        position_ids=pos,
        noise_embedding=emb,
        target_hidden_states=target_hidden,
        attention_mask=block_mask,
    )
    base = draft.compute_logits(h).reshape(1, 1, K, -1).float()

    def nll(logits4d):
        lp = torch.log_softmax(logits4d.reshape(K, -1), -1)
        return [(-lp[k, gt[k]]).item() for k in range(K)]

    out = {"L_D_base": nll(base)}

    mk = getattr(draft, "markov_head", None)
    if mk is None:
        out["L_D_mk0"] = out["L_D_base"]
        out["L_D_mk_tf"] = out["L_D_base"]
        return out

    # slot-0 only: the anchor is the one previous token known at serve time.
    prev_known = torch.full((1, 1, K), mask_id, dtype=torch.long, device=dev)
    prev_known[:, :, 0] = anchor_tok
    mk0 = mk.apply_block_logits(base.clone(), token_ids=prev_known,
                                hidden_states=None)
    slot0 = nll(mk0)
    out["L_D_mk0"] = [slot0[0]] + out["L_D_base"][1:]

    # teacher-forced: [anchor, gt_0 .. gt_{K-2}]  (modeling.py:607-615)
    prev_tf = torch.cat([
        torch.tensor([[[anchor_tok]]], device=dev),
        gt[: K - 1].reshape(1, 1, K - 1),
    ], dim=-1)
    out["L_D_mk_tf"] = nll(
        mk.apply_block_logits(base.clone(), token_ids=prev_tf, hidden_states=None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--corpus", required=True, choices=sorted(C.CORPORA),
                    help="stamped into the output so stats.py can refuse "
                         "cross-corpus joins")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--mask-id", type=int, default=151669)
    ap.add_argument("--target-layer-ids", default="1,9,17,25,33")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.corpus in C.PART_I_ONLY:
        raise SystemExit(
            f"{args.corpus} is a target-only contract: the drafter "
            f"was trained thinking-off, so a drafter NLL here measures transfer, "
            f"not context depth.")

    K = C.GAMMA
    tlids = [int(x) for x in args.target_layer_ids.split(",")]

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
        anchors = anchors[: args.limit]

    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    draft = _deepspec.Qwen3DSparkModel.from_pretrained(
        args.draft, dtype=torch.bfloat16, trust_remote_code=True
    ).to(target.device).eval()
    check_checkpoint(draft, args.mask_id, tlids, K)
    print(f"checkpoint validated: block={K} mask={args.mask_id} layers={tlids} "
          f"markov={'yes' if getattr(draft,'markov_head',None) else 'no'}",
          flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with out.open("w") as fout, torch.no_grad():
        for i, a in enumerate(anchors):
            seq = seqs[a["prompt_id"]]
            full = seq["prompt_ids"] + seq["response_ids"]
            cut = seq["prompt_len"] + a["t"]
            if cut < 1 or cut + K > len(full):
                continue
            prefix, gt = full[:cut], full[cut: cut + K]

            ids = torch.tensor([prefix], device=target.device)
            th = _deepspec.extract_context_feature(
                target(ids, output_hidden_states=True, use_cache=False).hidden_states,
                tlids,
            ).to(torch.bfloat16)

            res = block_nll(draft, th, len(prefix), prefix[-1],
                            torch.tensor(gt, device=target.device),
                            args.mask_id, K)

            fout.write(json.dumps({
                **{q: a[q] for q in ("prompt_id", "t", "context", "stratum", "pi")},
                "corpus": args.corpus,
                "draft": args.draft,
                **res,
            }) + "\n")
            n_written += 1

            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(anchors)}", flush=True)

    print(f"== {n_written} anchors -> {out}", flush=True)
    print("   L_D_base is the decomposition estimand (path-marginal, matches CE_B).")
    print("   L_D_mk_tf is an optimistic ceiling only -- it sees the in-block path.")


if __name__ == "__main__":
    main()

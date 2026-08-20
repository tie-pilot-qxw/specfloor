"""Numerically verify the sglang backend against transformers.

Reading sglang's source establishes what the logprob API is SUPPOSED to do. This
establishes what it actually does, on this build, for this model, at the exact
call shape the probes use. It is the arbiter for three things that would
otherwise be assumed:

  1. INDEX ALIGNMENT. Whether `input_token_logprobs` entry n corresponds to
     seq[start_len + n]. An off-by-one here shifts every CE by one slot and
     nothing downstream would look wrong.
  2. RAW vs TEMPERED. Whether the returned values are log_softmax of the
     unmodified logits. Run this with --corpus C1 (T=0.7/top_p=0.8/top_k=20) --
     that is the configuration where a tempered read would show up, and where a
     C0-only check would pass while the real run was silently wrong.
  3. THE token_ids_logprob PATH. Whether the input-side variant is populated at
     all, and whether its rows carry the ids we asked for in a usable form.

Run in three phases so transformers and sglang never share a CUDA context:

  python -m specfloor.verify_backend --phase hf   --corpus-file <f> --out /tmp/v
  python -m specfloor.verify_backend --phase sgl  --corpus-file <f> --out /tmp/v
  python -m specfloor.verify_backend --phase cmp  --out /tmp/v

Exit code is non-zero if any check fails, so it can gate the real run.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


from specfloor import config as C

# What is actually being checked, and why it is checked on dCE.
#
# Two bf16 stacks (HF eager vs sglang/fa3) do not agree to machine precision on
# a 151k-way softmax, so a per-token probability check has no well-defined pass
# threshold: at p ~ 1e-15 a 1% relative error is 0.4 nats and means nothing,
# while at p ~ 0.9 a 0.09-nat gap is a real 9%.
#
# The published quantities are CE_B and dCE = CE_B - CE_A, and dCE is far better
# conditioned than either term: both come from the same kernels on the same
# prefix, so the common-mode part of the error cancels. INFORMATIVE_THRESHOLD is
# 0.01 nats, so dCE is the number that has to be trustworthy -- and it is the
# one this file gates on. CE_A and CE_B are reported for information.
TOL_DCE = 0.01          # nats; equals INFORMATIVE_THRESHOLD
TOL_CE = 0.05           # nats; per-term, informational
M_PATHS = 16


def pick_cases(corpus_file, n, K, seed=C.SEED):
    """A handful of (prefix, gt) pairs spanning short and long contexts."""
    import random
    seqs = [json.loads(l) for l in open(corpus_file) if l.strip()]
    rng = random.Random(seed)
    cases = []
    for s in sorted(seqs, key=lambda d: d["prompt_len"] + d["response_len"]):
        full = s["prompt_ids"] + s["response_ids"]
        if s["response_len"] < 2 * K + 2:
            continue
        t = rng.randrange(1, s["response_len"] - K)
        cut = s["prompt_len"] + t
        cases.append({"prompt_id": s["prompt_id"], "t": t,
                      "prefix": full[:cut], "gt": full[cut: cut + K]})
    if not cases:
        raise SystemExit("no usable sequences in the corpus file")
    # spread across the length range rather than taking the head
    step = max(1, len(cases) // n)
    return cases[::step][:n]


# --------------------------------------------------------------- phase hf ---
def phase_hf(args):
    import torch
    from transformers import AutoModelForCausalLM

    K = C.GAMMA
    cases = pick_cases(args.corpus_file, args.cases, K)
    # .to() rather than device_map=: device_map pulls in accelerate, which the
    # sglang env does not carry, and this phase must be runnable in the SAME env
    # as the sglang phase or the comparison proves nothing about that env.
    model = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16).to("cuda").eval()

    out = []
    with torch.no_grad():
        for c in cases:
            prefix, gt = c["prefix"], c["gt"]
            tf = torch.tensor([prefix + gt], device=model.device)
            lp = torch.log_softmax(
                model(tf, logits_to_keep=K + 1).logits[0, :-1].float(), -1)
            ce_a = [(-lp[k, gt[k]]).item() for k in range(K)]

            # REALISTIC paths, sampled from the model at T=1, then RECORDED so
            # the sglang phase scores the identical conditioning states. An
            # arbitrary scrambled path would put p(gt_k) at ~1e-15, where a bf16
            # relative wobble is huge in nats and contributes nothing to any
            # mean -- i.e. it would test a regime the probe never visits.
            g = torch.Generator(device=model.device).manual_seed(C.SEED)
            cur = torch.tensor([prefix] * M_PATHS, device=model.device)
            o = model(cur, use_cache=True, logits_to_keep=1)
            past, logits = o.past_key_values, o.logits[:, -1].float()
            paths = [[] for _ in range(M_PATHS)]
            for _ in range(K - 1):
                nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
                for r in range(M_PATHS):
                    paths[r].append(int(nxt[r, 0]))
                o = model(nxt, use_cache=True, past_key_values=past)
                past, logits = o.past_key_values, o.logits[:, -1].float()

            probs = []
            for path in paths:
                seq = prefix + path + [gt[K - 1]]
                t2 = torch.tensor([seq], device=model.device)
                lp2 = torch.log_softmax(
                    model(t2, logits_to_keep=K + 1).logits[0, :-1].float(), -1)
                # row k predicts position len(prefix)+k = p(.|prefix, path[:k])
                probs.append([lp2[k, gt[k]].exp().item() for k in range(K)])

            ce_b = [-math.log(max(sum(p[k] for p in probs) / len(probs), 1e-300))
                    for k in range(K)]
            out.append({**{x: c[x] for x in ("prompt_id", "t")},
                        "prefix_len": len(prefix), "gt": gt, "paths": paths,
                        "ce_A": ce_a, "ce_B": ce_b,
                        "dCE": [b - a for b, a in zip(ce_b, ce_a)]})
    _write(args.out, "hf", {"cases": out})
    print(f"hf: wrote {len(out)} cases, {M_PATHS} sampled paths each")


# -------------------------------------------------------------- phase sgl ---
def phase_sgl(args):
    from specfloor import backend as B

    ref = _read(args.out, "hf")
    K = C.GAMMA
    cases = ref["cases"]
    max_ctx = max(c["prefix_len"] for c in cases) + 4 * K

    got = []
    with B.TargetEngine(args.target,
                        mem_fraction_static=B.resolve_mem_fraction(args.mem_fraction, args.target),
                        context_length=max_ctx, seed=C.SEED) as eng:
        # phase hf stored only the prefix LENGTH, so rebuild the token ids from
        # the corpus file; the assert below catches any selection drift.
        cases_full = pick_cases(args.corpus_file, args.cases, K)
        assert len(cases_full) == len(cases)
        for c, r in zip(cases_full, cases):
            assert c["prompt_id"] == r["prompt_id"] and c["t"] == r["t"], \
                "phase hf and phase sgl selected different anchors"
            prefix, gt = c["prefix"], c["gt"]
            paths = r["paths"]
            eng.warm_prefix(prefix)

            ce_a = eng.teacher_forced_nll([prefix + gt], [len(prefix)])[0]

            # the shipped route: token_ids_logprob over the same recorded paths
            seqs = [prefix + p + [gt[K - 1]] for p in paths]
            rows = eng.teacher_forced_query(seqs, [len(prefix)] * len(seqs),
                                            [list(gt)] * len(seqs))
            probs = [[math.exp(rw[k][gt[k]]) for k in range(K)] for rw in rows]

            # independent route using only input_token_logprobs, on the first
            # path. If the two sglang routes disagree, token_ids_logprob is
            # broken on this build and the probes must use the append route.
            aseq = [prefix + paths[0][:k] + [gt[k]] for k in range(K)]
            an = eng.teacher_forced_nll(aseq, [len(s) - 1 for s in aseq])
            q_app = [math.exp(-v[0]) for v in an]

            ce_b = [-math.log(max(sum(p[k] for p in probs) / len(probs), 1e-300))
                    for k in range(K)]
            got.append({"prompt_id": c["prompt_id"], "t": c["t"],
                        "ce_A": list(ce_a), "ce_B": ce_b,
                        "dCE": [b - a for b, a in zip(ce_b, ce_a)],
                        "p_path0": probs[0],
                        "p_path0_append": q_app})
    _write(args.out, "sgl", {"cases": got})
    print(f"sgl: wrote {len(got)} cases")


# -------------------------------------------------------------- phase cmp ---
def _spread(hf, sg, hkey, skey):
    ds = []
    for a, b in zip(hf["cases"], sg["cases"]):
        for x, y in zip(a[hkey], b[skey]):
            ds.append(float("inf") if not (math.isfinite(x) and math.isfinite(y))
                      else abs(x - y))
    ds.sort()
    return ds


def phase_cmp(args):
    hf = _read(args.out, "hf")
    sg = _read(args.out, "sgl")
    n_case = len(hf["cases"])
    print(f"\n{n_case} anchors x {C.GAMMA} slots, {M_PATHS} shared sampled paths")
    print("paths are RECORDED in phase hf and replayed in phase sgl, so both "
          "backends\nscore identical conditioning states.\n")

    ok = True
    rows = [("dCE = CE_B - CE_A", "dCE", "dCE", TOL_DCE, True),
            ("CE_A", "ce_A", "ce_A", TOL_CE, False),
            ("CE_B", "ce_B", "ce_B", TOL_CE, False)]
    dce_max = None
    for label, hk, sk, tol, gating in rows:
        ds = _spread(hf, sg, hk, sk)
        med, p90, mx = ds[len(ds) // 2], ds[int(0.9 * len(ds))], ds[-1]
        # Gate on p90, not the max. The max of a few dozen bf16 differences is
        # not a stable statistic -- it moves with the sample -- whereas p90 is.
        # The max is still printed, because it is the number the protocol's
        # equivalence delta has to respect.
        good = p90 <= tol
        if gating:
            ok &= good
            dce_max = mx
        tag = ("PASS" if good else "FAIL") if gating else ("ok" if good else "--")
        print(f"  [{tag:>4}] {label:<20} median {med:.5f}  p90 {p90:.5f}  "
              f"max {mx:.5f} nats   (p90 tol {tol})")

    # the two sglang routes must agree with EACH OTHER exactly-ish; if they do
    # not, token_ids_logprob is broken on this build regardless of what hf says
    dd = []
    for b in sg["cases"]:
        for x, y in zip(b["p_path0"], b["p_path0_append"]):
            m = max(x, y, 1e-300)
            dd.append(abs(x - y) / m)
    dd.sort()
    route_ok = dd[-1] <= 1e-3
    ok &= route_ok
    print(f"  [{'PASS' if route_ok else 'FAIL'}] token_ids_logprob vs append "
          f"route: max relative diff {dd[-1]:.2e}")

    print()
    if not ok:
        print("A gating failure means the probes would publish wrong numbers.")
        print("  * dCE off by a constant          -> index alignment")
        print("  * dCE skewed smoothly            -> temperature contamination")
        print("    (check SGLANG_RETURN_ORIGINAL_LOGPROB reached the workers)")
        print("  * the two sglang routes disagree -> token_ids_logprob is")
        print("    unusable here; switch the probes to the append route")
        sys.exit(1)
    print("backend verified: sglang reproduces the published estimands on this "
          "build.\nCE_A/CE_B may differ from HF by more than dCE does -- that is "
          "the common-mode\nbf16 kernel difference cancelling in the difference, "
          "which is why dCE gates.")
    # The max alone is misleading: it is driven by anchors with ENORMOUS dCE,
    # where an absolute gap of 0.3 nats is a fraction of a percent. What the
    # headline incidence actually depends on is whether an anchor lands on the
    # same side of INFORMATIVE_THRESHOLD in both stacks, so that is reported.
    thr = C.INFORMATIVE_THRESHOLD
    pairs = [(x, y) for a, b in zip(hf["cases"], sg["cases"])
             for x, y in zip(a["dCE"], b["dCE"])]
    flips = sum((x > thr) != (y > thr) for x, y in pairs)
    over = [(abs(x - y), x) for x, y in pairs if abs(x - y) > thr]
    worst_rel = max((d / abs(v) for d, v in over if abs(v) > 1e-9), default=0.0)
    print(f"\nCALIBRATION -- cross-backend agreement over {len(pairs)} values")
    print(f"  |ddCE| > {thr} nats            : {len(over)} "
          f"({len(over)/len(pairs):.1%})")
    print(f"  worst absolute |ddCE|          : {dce_max:.5f} nats")
    print(f"  worst RELATIVE among those     : {worst_rel:.1%}   "
          f"(large absolute gaps sit at large dCE)")
    print(f"  informative/not classification : {flips} flips at the "
          f"{thr}-nat threshold ({flips/len(pairs):.2%})")
    if flips == 0:
        print("  -> the incidence headline is stable across backends at this n.")
    else:
        print("  -> near-threshold anchors are backend-sensitive; quote "
              "incidence with this\n     flip rate, and do not read individual "
              "near-threshold anchors.")
    print(f"  PROTOCOL.md EQUIVALENCE_DELTA should be set no tighter than "
          f"{dce_max:.3f} nats.")


def _p(out, tag):
    return pathlib.Path(f"{out}.{tag}.json")


def _write(out, tag, obj):
    p = _p(out, tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _read(out, tag):
    p = _p(out, tag)
    if not p.exists():
        raise SystemExit(f"missing {p} -- run --phase {tag} first")
    return json.loads(p.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("hf", "sgl", "cmp"))
    ap.add_argument("--corpus-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="0 = size from free VRAM")
    args = ap.parse_args()
    if args.phase in ("hf", "sgl") and not args.corpus_file:
        raise SystemExit("--corpus-file is required for phases hf and sgl")
    {"hf": phase_hf, "sgl": phase_sgl, "cmp": phase_cmp}[args.phase](args)


if __name__ == "__main__":
    main()

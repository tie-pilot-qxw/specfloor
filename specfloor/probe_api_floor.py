"""T^(0) and T^(1) on a target that is only reachable through an API.

The floor is a property of the TARGET's conditional structure, not of any
drafter, so it is the one half of the floors-and-gaps decomposition that a
remote endpoint can supply. The other half cannot be had at any price: a block
drafter consumes the target's INTERMEDIATE HIDDEN STATES (taps [1,9,17,25,33]
for the Qwen3-4B checkpoints), and no serving API exposes activations. R, G and
exposure are therefore out of reach here, and this probe does not pretend to
estimate them.

WHAT MAKES THIS CHEAP, and why it does not need the operations backend_api.py
refuses. A single generation request with `max_tokens = gamma`, `temperature 1`,
`top_p 1` and `logprobs` returns, in one round trip:

  * one free rollout Z_0..Z_{gamma-1} drawn from the target's own law, and
  * the top-K distribution at EVERY slot, i.e. p(. | X, Z_{<k}) for all k,

because the reported per-position logprobs are conditioned on what was already
emitted. That is exactly the object `probe_tk` teacher-forces a path to obtain,
so rung 0 costs M requests per anchor rather than M*gamma, and no `echo` is
needed -- which matters, since this endpoint rejects `echo` together with
`logprobs`.

THE ORDER-1 FLOOR COMES FROM GROUPING, NOT SNIS. Forcing a chosen suffix onto a
sampled prefix requires teacher forcing, which the endpoint does not offer. But
T^(1) only needs the paths to be partitioned by their own realised Z_{k-1} and
minimised within each cell, and free rollouts are posterior-distributed inside a
cell by construction. Both the plug-in and split-half variants of probe_rpre are
reused verbatim so the estimator is identical to the local one.

TRUNCATION IS THE ONE THING TO CHECK BEFORE BELIEVING A NUMBER. The endpoint
caps top_logprobs at 20. A tail that a top-20 read cannot see is a tail the TV
cannot charge for, so every cell carries its measured residual 1 - sum of the
20 reported masses, and `--resid-gate` drops the cells where it is not
negligible against the quantity being estimated. Measured on this endpoint at
C0: median 4.8e-08, mean 6.4e-04, p99 8.5e-03, with chat an order worse than
code or math -- so the gate bites on a few percent of cells and is reported
rather than hidden.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import pathlib
import random
import time
import urllib.error
import urllib.request

import torch

from specfloor import config as C
from specfloor.probe_rpre import barycentre, cond_floor, mean_tv


# ------------------------------------------------------------------- client --
class API:
    """Minimal OpenAI-compatible client with retry, and usage accounting.

    Usage is tallied because the whole point of this probe is that a frontier
    target is affordable, and a claim like that has to come with the bill.
    """

    def __init__(self, base, model, key, retries=5):
        self.base = base.rstrip("/")
        self.model = model
        self.key = key
        self.retries = retries
        self.usage = dict(requests=0, prompt=0, cached=0, completion=0, retried=0)

    def _post(self, path, body):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + self.key,
                     "Content-Type": "application/json"})
        last = None
        for a in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=300) as f:
                    d = json.load(f)
                u = d.get("usage") or {}
                self.usage["requests"] += 1
                self.usage["prompt"] += int(u.get("prompt_tokens", 0))
                self.usage["cached"] += int(u.get("prompt_cache_hit_tokens", 0))
                self.usage["completion"] += int(u.get("completion_tokens", 0))
                return d
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError) as e:
                last = e
                self.usage["retried"] += 1
                time.sleep(min(30.0, 2.0 ** a))
        raise RuntimeError(f"{path} failed after {self.retries} tries: {last}")

    def chat(self, messages, max_tokens, top_logprobs, temperature=1.0, top_p=1.0):
        """A generation whose per-position top-K logprobs are the measurement.

        `thinking` is disabled explicitly: C0 is a non-thinking contract, and a
        reasoning model would otherwise put the block being measured inside a
        stream that never reaches the user.
        """
        body = dict(model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, logprobs=True,
                    top_logprobs=top_logprobs, thinking={"type": "disabled"})
        d = self._post("/chat/completions", body)
        ch = d["choices"][0]
        return ch["message"].get("content") or "", (ch.get("logprobs") or {}).get("content") or []


# -------------------------------------------------------------- estimators --
def slot_matrix(paths, k, device="cpu"):
    """[M_alive, |V_union|] target probabilities at slot k, plus bookkeeping.

    Tokens outside a path's own top-K are recorded as zero. That is the only
    approximation in this probe and it is bounded by that path's residual, which
    travels with the cell so the gate can act on it. Rows are renormalised to
    unit mass so the barycentre solver sees genuine distributions -- the
    alternative, sub-probability rows, makes TV silently read ~0.5 everywhere.
    """
    rows, cond, resid = [], [], []
    for p in paths:
        if k >= len(p):
            continue
        rows.append(p[k]["top"])
        resid.append(p[k]["resid"])
        cond.append(p[k - 1]["tok"] if k > 0 else "<anchor>")
    if len(rows) < 2:
        return None
    vocab = {}
    for r in rows:
        for t in r:
            if t not in vocab:
                vocab[t] = len(vocab)
    P = torch.zeros((len(rows), len(vocab)), dtype=torch.float64, device=device)
    for i, r in enumerate(rows):
        for t, v in r.items():
            P[i, vocab[t]] = v
    P = P / P.sum(dim=1, keepdim=True).clamp(min=1e-300)
    cmap = {}
    cid = torch.tensor([cmap.setdefault(c, len(cmap)) for c in cond], device=device)
    return P, cid, sum(resid) / len(resid), len(vocab)


def anchor_floor(paths, gamma, device="cpu"):
    out = {}
    for k in range(gamma):
        got = slot_matrix(paths, k, device)
        if got is None:
            continue
        P, cid, resid, nv = got
        cell = dict(n=int(P.shape[0]), resid=resid, support=nv,
                    T0=mean_tv(P, barycentre(P)))
        cell["T1"] = cond_floor(P, cid)
        # Split-half on T^(0) too: the barycentre is fitted on these very paths,
        # so the in-sample number is optimistic by construction and the size of
        # that optimism is worth carrying rather than assuming away.
        idx = torch.randperm(P.shape[0], device=device)
        h1, h2 = idx[0::2], idx[1::2]
        if h1.numel() >= 2 and h2.numel() >= 2:
            cell["T0_split"] = mean_tv(P[h2], barycentre(P[h1]))
        out[str(k)] = cell
    return out


# -------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="https://api.deepseek.com/beta")
    ap.add_argument("--api-model", default="deepseek-v4-pro")
    ap.add_argument("--prompts", required=True,
                    help="jsonl with a 'prompt' text field and optional 'domain'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", type=int, default=32, help="anchors in total")
    ap.add_argument("--paths", type=int, default=64, help="M free rollouts per anchor")
    ap.add_argument("--gamma", type=int, default=C.GAMMA)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--response-tokens", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-requests", type=int, default=100000,
                    help="hard stop, so a misconfigured run cannot spend the budget")
    args = ap.parse_args()

    key = os.environ.get("MEASUREMENT_API_KEY")
    if not key:
        raise SystemExit("set MEASUREMENT_API_KEY")
    api = API(args.api_base, args.api_model, key)
    rng = random.Random(C.SEED)

    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]
    print(f"== {len(prompts)} prompts; target {args.api_model} via {args.api_base}",
          flush=True)
    print(f"== C0 contract: temperature 1, top_p 1, thinking disabled, top-{args.top_k} read",
          flush=True)

    # ---- stage 1: the target writes its own responses under C0. The floor is a
    # property of THIS target's conditional structure, so it has to be measured
    # on text this target actually produces, not on another model's.
    def gen(p):
        txt, toks = api.chat([{"role": "user", "content": p["prompt"]}],
                             args.response_tokens, args.top_k)
        return p, txt, toks

    corpus = []
    with cf.ThreadPoolExecutor(args.concurrency) as ex:
        for p, txt, toks in ex.map(gen, prompts):
            if len(toks) >= args.gamma + 8:
                corpus.append((p, [t["token"] for t in toks]))
    print(f"== {len(corpus)} responses kept ({args.response_tokens} max tokens)", flush=True)
    if not corpus:
        raise SystemExit("no usable responses")

    # ---- stage 2: anchors, stratified over relative position exactly as the
    # local runs are, so the two are comparable cell for cell.
    # A block at t=0 has no response context at all and a block that starts
    # within gamma of the end runs into EOS on every path, losing the deep slots
    # that carry most of the floor. Both are excluded by construction rather
    # than filtered afterwards, since either would bias the pooled slot means.
    LEAD, TAIL = 8, args.gamma + 4
    anchors = []
    for p, toks in corpus:
        lo, hi = LEAD, len(toks) - TAIL
        if hi <= lo:
            continue
        for lo_f, hi_f in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0)):
            a = lo + int(lo_f * (hi - lo))
            b = max(lo + int(hi_f * (hi - lo)) - 1, a)
            anchors.append((p, toks, rng.randint(a, b)))
    rng.shuffle(anchors)
    anchors = anchors[: args.anchors]
    print(f"== {len(anchors)} anchors over {len({id(a[1]) for a in anchors})} responses; "
          f"M={args.paths}; gamma={args.gamma}", flush=True)

    need = len(anchors) * args.paths
    if need > args.max_requests:
        raise SystemExit(f"{need} rollout requests exceeds --max-requests {args.max_requests}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with out.open("w") as fout:
        for ai, (p, toks, t) in enumerate(anchors):
            prefix = "".join(toks[:t])
            msgs = [{"role": "user", "content": p["prompt"]},
                    {"role": "assistant", "content": prefix, "prefix": True}]

            def one(_):
                _, lp = api.chat(msgs, args.gamma, args.top_k)
                path = []
                for pos in lp:
                    tops = {d["token"]: math.exp(d["logprob"]) for d in pos["top_logprobs"]}
                    path.append(dict(tok=pos["token"], top=tops,
                                     resid=max(0.0, 1.0 - sum(tops.values()))))
                return path

            # One request first, alone, so the shared prefix lands in the
            # endpoint's cache before the fan-out. Firing all M at once makes
            # every one of them a cache miss and multiplies the prompt bill by M.
            paths = [one(0)]
            with cf.ThreadPoolExecutor(args.concurrency) as ex:
                paths += [q for q in ex.map(one, range(args.paths - 1)) if q]

            row = dict(prompt_id=p.get("prompt_id", f"p{ai}"), domain=p.get("domain", "?"),
                       t=t, context=t, M=len(paths), K=args.gamma, top_k=args.top_k,
                       corpus="C0", model=args.api_model,
                       slots=anchor_floor(paths, args.gamma))
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            s = row["slots"].get(str(args.gamma - 1)) or {}
            t1 = (s.get("T1") or {}).get("split")
            print(f"   {ai+1}/{len(anchors)} {row['domain']:8s} t={t:4d} "
                  f"deep n={s.get('n', 0):4d} T0={s.get('T0', float('nan')):.4f} "
                  f"T1={'--' if t1 is None else '%.4f' % t1} "
                  f"resid={s.get('resid', 0):.1e} | {api.usage['requests']} reqs "
                  f"{100*api.usage['cached']/max(api.usage['prompt'],1):.0f}% cached "
                  f"{time.time()-t0:.0f}s", flush=True)

    u = api.usage
    print(f"\n== wrote {len(anchors)} rows to {out} in {time.time()-t0:.0f}s")
    print(f"== usage: {u['requests']} requests, {u['prompt']} prompt tokens "
          f"({u['cached']} cache hits, {100*u['cached']/max(u['prompt'],1):.0f}%), "
          f"{u['completion']} completion tokens, {u['retried']} retries")
    print("T is the floor for the TARGET. R, G and exposure need the drafter, which")
    print("needs the target's hidden-state taps -- unavailable through any endpoint.")


if __name__ == "__main__":
    main()

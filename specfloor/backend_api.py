"""Remote-endpoint backend, for targets too large to host on the local GPUs.

Same three primitives as backend.TargetEngine, so the probes do not care which
one they are given. What differs is that a remote endpoint may not EXPOSE the
operations the measurement needs, and the failure mode is silent: a chat API
will happily return logprobs that are the wrong quantity.

-------------------------------------------------------------------------------
CAPABILITY TIERS. Probed empirically at startup, never assumed from the URL.

  TIER A  native sglang /generate with token_ids_logprob
          -> identical to the local backend. Full fidelity.

  TIER B  OpenAI-compatible /v1/completions with echo=true + logprobs=N,
          accepting TOKEN IDS as `prompt`
          -> prompt logprobs available, so CE_A is exact in one request and
             CE_B is one request per path. p(gt_k) is exact whenever gt_k is
             inside the returned top-N, which for CE_A is guaranteed (the
             echoed token is always reported) and for CE_B is not.

  TIER C  chat-only, top_logprobs <= 20, no echo
          -> REFUSED by default. See the note below; this is not a matter of
             writing more code.

-------------------------------------------------------------------------------
WHY TIER C IS REFUSED

Two independent problems, and the second is fatal to the story rather than to
the engineering.

1. Request multiplier. Without echo there is no way to score a token in place,
   so CE_A needs one request per slot (send prefix+gt[:k], read the next-token
   distribution) and CE_B needs one per (path, slot). At M=128, gamma=7 that is
   ~900 requests per anchor, ~2.2M per domain at the 2500-anchor budget. No
   commercial endpoint makes that affordable, and rate limits make it slow even
   if it were.

2. Informative-dependent censoring. top_logprobs caps at 20, so p(gt_k) is
   observable only when gt_k is among the 20 most likely continuations. The
   anchors that carry this paper's entire result are exactly the ones where the
   realized token is improbable under a wrong path -- i.e. where gt_k is NOT in
   the top 20. The censoring is therefore correlated with the quantity being
   measured, and the observed subset is biased toward the uninformative anchors.

   The censoring can be turned into rigorous INTERVAL bounds -- a missing p is
   known to lie in [0, p_20th], which brackets CE_B and hence dCE -- but that
   changes every downstream estimand from a number to an interval, and stats.py,
   the bootstrap and the incidence definition would all have to be rewritten
   around interval arithmetic. That is a real project, not a flag, and it is not
   started. `--allow-degraded` therefore only relaxes the tier check for
   SAMPLING (corpus generation), which is unaffected by any of this.

-------------------------------------------------------------------------------
So the practical routes for a large target are, in order of preference:

  * rent a GPU host and run `sglang.launch_server` on it -> Tier A, and the
    measurement is unchanged;
  * a provider that exposes /v1/completions with echo+logprobs over raw token
    ids (vLLM- or sglang-backed) -> Tier B;
  * an official chat API -> corpus generation only.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request

from specfloor import config as C

TIER_A, TIER_B, TIER_C = "A", "B", "C"


class Capabilities:
    def __init__(self, tier, top_logprobs=0, detail=""):
        self.tier, self.top_logprobs, self.detail = tier, top_logprobs, detail

    def __repr__(self):
        return f"<Tier {self.tier}, top_logprobs={self.top_logprobs}: {self.detail}>"


def _post(url, payload, key=None, timeout=120, retries=3):
    body = json.dumps(payload).encode()
    hdr = {"Content-Type": "application/json"}
    if key:
        hdr["Authorization"] = f"Bearer {key}"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode(errors="replace")
            last = RuntimeError(f"HTTP {e.code} from {url}: {detail}")
            if e.code in (400, 401, 403, 404, 422):
                raise last            # a contract error; retrying cannot help
        except Exception as e:        # noqa: BLE001 - transport, retry
            last = e
        time.sleep(2 ** attempt)
    raise last


# ------------------------------------------------------------------ probe ---
def probe_capabilities(base_url: str, model: str, key: str | None,
                       sample_ids=None) -> Capabilities:
    """Establish what this endpoint can actually do, by asking it.

    Never inferred from the hostname: a URL that looks like sglang may be a
    proxy, and an OpenAI-compatible path may be served by anything.
    """
    ids = list(sample_ids or [9906, 1917, 0, 358, 1097])
    base = base_url.rstrip("/")

    # Reachability FIRST. Without this, a wrong URL or a dead server comes back
    # as "Tier C", i.e. "your endpoint cannot do this measurement" -- and
    # someone re-plans a run around a typo. Absence of capability and absence of
    # a server are different findings and must not share a code path.
    reachable = False
    for path, payload in ((f"{base}/v1/models", None),
                          (f"{base}/get_model_info", None),
                          (f"{base}/health", None)):
        try:
            req = urllib.request.Request(path)
            if key:
                req.add_header("Authorization", f"Bearer {key}")
            urllib.request.urlopen(req, timeout=15).read()
            reachable = True
            break
        except urllib.error.HTTPError:
            reachable = True          # it answered, just not with 200
            break
        except Exception:             # noqa: BLE001 - transport
            continue
    if not reachable:
        raise SystemExit(
            f"cannot reach {base} -- no response from /v1/models, "
            f"/get_model_info or /health.\nThis is a connectivity or URL "
            f"problem, NOT a capability tier. Fix the endpoint before "
            f"concluding anything about what it supports.")

    # --- Tier A: native sglang generate with token_ids_logprob --------------
    try:
        r = _post(f"{base}/generate", {
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "token_ids_logprob": ids[:2],
        }, key, retries=1)
        meta = (r[0] if isinstance(r, list) else r).get("meta_info", {})
        if meta.get("input_token_logprobs") and meta.get("input_token_ids_logprobs"):
            return Capabilities(TIER_A, 0,
                                "native /generate with input-side token_ids_logprob")
    except Exception as e:      # noqa: BLE001 - absence is the normal case
        a_err = str(e)[:120]
    else:
        a_err = "responded but without input-side logprob fields"

    # --- Tier B: /v1/completions with echo + logprobs over token ids --------
    try:
        r = _post(f"{base}/v1/completions", {
            "model": model, "prompt": ids, "max_tokens": 0,
            "echo": True, "logprobs": 5, "temperature": 1.0,
        }, key, retries=1)
        lp = (r.get("choices") or [{}])[0].get("logprobs") or {}
        tl = lp.get("token_logprobs") or []
        # echo works only if it scored the PROMPT: >=2 entries, first is null
        if len(tl) >= len(ids) - 1 and any(x is not None for x in tl):
            n_top = max((len(d) for d in (lp.get("top_logprobs") or []) if d),
                        default=0)
            return Capabilities(TIER_B, n_top,
                                f"/v1/completions echo+logprobs over token ids "
                                f"(top_logprobs={n_top})")
        b_err = f"echo returned {len(tl)} token_logprobs for a {len(ids)}-token prompt"
    except Exception as e:      # noqa: BLE001
        b_err = str(e)[:120]

    return Capabilities(TIER_C, 20,
                        f"no prompt-logprob path. tierA: {a_err} | tierB: {b_err}")


# ------------------------------------------------------------------ target --
class APITarget:
    """Drop-in for backend.TargetEngine against a remote endpoint."""

    def __init__(self, base_url: str, model: str, key: str | None = None,
                 allow_degraded: bool = False, timeout: int = 120):
        self.base = base_url.rstrip("/")
        self.model = model
        self.key = key or os.environ.get("MEASUREMENT_API_KEY")
        self.timeout = timeout
        self.caps = probe_capabilities(self.base, model, self.key)
        print(f"endpoint capability: {self.caps}", flush=True)

        if self.caps.tier == TIER_C and not allow_degraded:
            raise SystemExit(
                "This endpoint exposes no prompt-logprob path, so CE_A and CE_B "
                "cannot be measured\nat full fidelity. See the module docstring: "
                "the blocker is informative-dependent\ncensoring at "
                "top_logprobs<=20, not missing code. Options:\n"
                "  * run sglang.launch_server on a rented host (Tier A, "
                "measurement unchanged)\n"
                "  * use a /v1/completions endpoint with echo+logprobs (Tier B)\n"
                "  * pass --allow-degraded to use this endpoint for CORPUS "
                "GENERATION only")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def warm_prefix(self, prefix) -> None:
        """No-op. Prefix reuse is the server's business and is not observable
        from here; the local backend's warm-up exists to beat a cold radix tree
        in the SAME process, which does not transfer."""
        return

    # ------------------------------------------------------------------ 1 --
    def generate_ids(self, prompts_ids, policy, max_new_tokens,
                     stop_token_ids=None, ignore_eos=False, seed=None):
        """Batched free generation. Works at EVERY tier -- sampling needs no
        logprob support, which is why --allow-degraded permits it.

        `seed` may be a scalar or one value per prompt; corpus.py needs the
        latter, because its whole reproducibility story is a per-prompt seed
        that does not depend on batch composition.
        """
        seeds = (list(seed) if isinstance(seed, (list, tuple))
                 else [seed] * len(prompts_ids))
        if len(seeds) != len(prompts_ids):
            raise ValueError(f"{len(seeds)} seeds for {len(prompts_ids)} prompts")
        out = []
        for ids, seed in zip(prompts_ids, seeds):
            body = {"model": self.model, "prompt": list(ids),
                    "max_tokens": int(max_new_tokens),
                    "temperature": float(policy["temp"]),
                    "top_p": float(policy["top_p"])}
            if policy.get("top_k"):
                body["top_k"] = int(policy["top_k"])
            if stop_token_ids:
                body["stop_token_ids"] = list(stop_token_ids)
            if seed is not None:
                body["seed"] = int(seed)
            r = _post(f"{self.base}/v1/completions", body, self.key, self.timeout)
            ch = (r.get("choices") or [{}])[0]
            toks = ch.get("token_ids") or ch.get("tokens") or []
            out.append(([int(t) for t in toks], ch.get("finish_reason")))
        return out

    # ------------------------------------------------------------------ 2 --
    def teacher_forced_nll(self, seqs, start_lens):
        """-log p(seq[j] | seq[:j]) for j >= start_len, per sequence."""
        if self.caps.tier == TIER_A:
            return self._nll_native(seqs, start_lens)
        if self.caps.tier == TIER_B:
            return self._nll_echo(seqs, start_lens)
        raise SystemExit("teacher_forced_nll needs Tier A or B")

    def _nll_native(self, seqs, start_lens):
        from specfloor.backend import _align_input_logprobs

        res = []
        for seq, s in zip(seqs, start_lens):
            r = _post(f"{self.base}/generate", {
                "input_ids": list(seq),
                "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
                "return_logprob": True,
                "logprob_start_len": max(0, s - 1),
            }, self.key, self.timeout)
            meta = (r[0] if isinstance(r, list) else r)["meta_info"]
            res.append(_align_input_logprobs(meta["input_token_logprobs"], seq, s))
        return res

    def _nll_echo(self, seqs, start_lens):
        """OpenAI echo convention: token_logprobs[j] is the logprob of prompt
        token j, and entry 0 is null (no predecessor). Aligned by index, and the
        length is asserted rather than trusted."""
        res = []
        for seq, s in zip(seqs, start_lens):
            r = _post(f"{self.base}/v1/completions", {
                "model": self.model, "prompt": list(seq), "max_tokens": 0,
                "echo": True, "logprobs": 1, "temperature": 1.0,
            }, self.key, self.timeout)
            tl = (r["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
            if len(tl) != len(seq):
                raise RuntimeError(
                    f"echo returned {len(tl)} logprobs for a {len(seq)}-token "
                    f"prompt; the alignment convention does not hold here")
            res.append([-float(tl[j]) if tl[j] is not None else float("nan")
                        for j in range(s, len(seq))])
        return res

    # ------------------------------------------------------------------ 3 --
    def teacher_forced_query(self, seqs, start_lens, query_ids):
        """logprobs of ARBITRARY ids at each scored position.

        Tier A only. Tier B's echo returns the logprob of the token that is
        actually there plus a top-N list; the ids we need are generally not in
        that list precisely at the anchors that matter, which is the same
        censoring problem that disqualifies Tier C -- just with a different N.
        """
        if self.caps.tier != TIER_A:
            raise SystemExit(
                f"teacher_forced_query needs arbitrary-token logprobs, which "
                f"this endpoint (Tier {self.caps.tier}) does not expose.\n"
                f"CE_B is therefore not measurable here. Tier B can still do "
                f"CE_A; run the CE_B half on a Tier A endpoint.")
        from specfloor.backend import _align_input_query

        res = []
        for seq, s, q in zip(seqs, start_lens, query_ids):
            r = _post(f"{self.base}/generate", {
                "input_ids": list(seq),
                "sampling_params": {"max_new_tokens": 0, "temperature": 1.0},
                "return_logprob": True,
                "logprob_start_len": max(0, s - 1),
                "token_ids_logprob": list(q),
            }, self.key, self.timeout)
            meta = (r[0] if isinstance(r, list) else r)["meta_info"]
            res.append(_align_input_query(meta["input_token_ids_logprobs"],
                                          seq, s, q))
        return res


def make_target(args, need_logprobs: bool = True):
    """Return an APITarget when --api-base is given, else a local TargetEngine.

    `need_logprobs=False` is passed by corpus generation, which is the one stage
    that works at every tier.
    """
    if getattr(args, "api_base", None):
        t = APITarget(args.api_base, args.api_model or "default",
                      args.api_key, allow_degraded=not need_logprobs
                      or getattr(args, "allow_degraded", False))
        if need_logprobs and t.caps.tier != TIER_A:
            raise SystemExit(
                f"this stage needs arbitrary-token logprobs (Tier A); the "
                f"endpoint probed as Tier {t.caps.tier}.\n{t.caps.detail}")
        return t

    from specfloor import backend as B

    return B.TargetEngine(
        args.target,
        mem_fraction_static=B.resolve_mem_fraction(args.mem_fraction, args.target),
        context_length=getattr(args, "context_length", None),
        max_running_requests=getattr(args, "max_running", None),
        seed=C.SEED)


def add_api_args(ap):
    ap.add_argument("--api-base", default=None,
                    help="remote endpoint; if set, the probes use it instead of "
                         "a local sglang Engine")
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--api-key", default=None,
                    help="or set MEASUREMENT_API_KEY")
    ap.add_argument("--allow-degraded", action="store_true",
                    help="permit a Tier C endpoint for CORPUS GENERATION only")
    return ap


def main() -> None:
    """Standalone capability check -- run this before planning a remote run."""
    import argparse

    ap = argparse.ArgumentParser(description="probe a remote endpoint")
    add_api_args(ap)
    args = ap.parse_args()
    if not args.api_base:
        raise SystemExit("--api-base is required")
    caps = probe_capabilities(args.api_base, args.api_model or "default",
                              args.api_key or os.environ.get("MEASUREMENT_API_KEY"))
    print(f"\n{caps}\n")
    verdict = {
        TIER_A: "FULL: CE_A, CE_B, R_m and corpus generation all work unchanged.",
        TIER_B: "PARTIAL: CE_A and corpus generation work. CE_B/R_m do NOT -- "
                "they need\n  arbitrary-token logprobs. Split the run across "
                "endpoints, or use Tier A.",
        TIER_C: "CORPUS ONLY: no prompt-logprob path. See the module docstring "
                "for why this is\n  a measurement limit rather than a missing "
                "feature.",
    }[caps.tier]
    print(verdict)


if __name__ == "__main__":
    main()

"""sglang offline-Engine backend for the measurement probes.

Why sglang and not transformers
-------------------------------
The probes need M sampled continuations of the SAME prefix. Under transformers
that costs M independent KV caches: on Qwen3-4B the KV is

    36 layers x 2 x 8 kv-heads x 128 head_dim x 2 bytes = 144 KiB / token / path

so M=128 paths over an 8k prefix is 8000 * 128 * 144KiB = 141 GiB. That does not
fit, which is why the transformers version had to chunk the path batch and
re-prefill the prefix once per chunk -- quadratic busywork to work around a
problem that only existed because the prefix was being copied.

sglang's RadixAttention stores the shared prefix ONCE and only the divergent
suffixes separately, so the same job costs

    8000 + 128 * 7 = 8896 token-slots ~ 1.25 GiB

i.e. it fits in the scraps of a shared GPU. Prefix reuse is also what makes the
second (scoring) pass below nearly free.

The scoring contract
--------------------
PROTOCOL.md requires probabilities read from the RAW head: no temperature, no
truncation. sglang does NOT do this by default. In srt/layers/sampler.py:189 the
standard decode path runs

    logits.div_(sampling_info.temperatures)      # in-place
    logits[:] = torch.softmax(logits, dim=-1)
    logprobs = torch.log(probs)                  # <-- temperature-scaled

so OUTPUT logprobs are tempered unless SGLANG_RETURN_ORIGINAL_LOGPROB=1
(sampler.py:150,232; default False at srt/environ.py:907). Under C1
(T=0.7/top_p=0.8/top_k=20) that would have silently rescaled every probability we
report. INPUT (prefill) logprobs are unaffected -- they are a plain log_softmax
of the raw logits (logprob_processor.py:504,733), because the temperature lives
in the Sampler, which only ever touches next_token_logits.

This module therefore does two things:
  1. sets SGLANG_RETURN_ORIGINAL_LOGPROB=1 before the engine is constructed, and
  2. reads every probability from the INPUT side, via teacher forcing, so the
     numbers are raw by construction and not merely by flag.

Sampling and scoring are consequently SEPARATE passes. That is deliberate: it
costs one extra prefill of gamma radix-cached tokens per path and buys
independence from the sampler's logprob conventions entirely.
"""

from __future__ import annotations

import os

# Must precede `import sglang`: srt/layers/sampler.py reads this at import time
# into a module-level constant (sampler.py:65), so setting it later is a no-op.
os.environ["SGLANG_RETURN_ORIGINAL_LOGPROB"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import math

_ENGINE_SINGLETON = None


def _check_env() -> None:
    """Fail loudly if the raw-logprob flag did not take."""
    from sglang.srt.layers import sampler as _s

    if not getattr(_s, "SGLANG_RETURN_ORIGINAL_LOGPROB", False):
        raise SystemExit(
            "SGLANG_RETURN_ORIGINAL_LOGPROB did not take effect. sglang was "
            "imported before specfloor.backend. Import this module first, or "
            "export the variable in the shell. Continuing would report "
            "temperature-scaled probabilities as if they were raw."
        )


# sglang asserts these architectures out of fa3 entirely
# (server_args.py:5478-5492), while deterministic mode separately requires a
# backend in RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND
# (= ascend/fa3/fa4/triton) or it drops the radix cache. triton is the only
# member of both sets, so it is what a Gemma4 target has to run on here.
_GEMMA4_ARCHS = (
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
)


def _attention_backend(model_path: str) -> str:
    """fa3 on Hopper, except where the architecture is not allowed to use it.

    Resolved from the config rather than passed in by the caller: a probe should
    not have to know which family its target belongs to, and getting this wrong
    is not a crash but a silent loss of the radix cache.
    """
    try:
        from transformers import AutoConfig

        archs = getattr(AutoConfig.from_pretrained(model_path), "architectures", None)
    except Exception:
        return "fa3"   # unreadable config: let sglang raise its own error
    archs = archs or []
    if any(a in _GEMMA4_ARCHS or str(a).startswith("Gemma4") for a in archs):
        return "triton"
    return "fa3"


class TargetEngine:
    """Thin wrapper over sglang's in-process Engine.

    Only three operations are exposed, and every probe is written in terms of
    them so there is exactly one place where the sglang API is interpreted.
    """

    def __init__(self, model_path: str, mem_fraction_static: float = 0.35,
                 context_length: int | None = None, dtype: str = "bfloat16",
                 max_running_requests: int | None = None,
                 seed: int = 0, deterministic: bool = True, **kw):
        global _ENGINE_SINGLETON
        _check_env()
        from sglang.srt.entrypoints.engine import Engine

        args = dict(
            model_path=model_path,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            skip_tokenizer_init=True,   # we work in token ids end to end
            disable_radix_cache=False,  # explicit: prefix sharing is the point
            random_seed=seed,
            log_level="warning",
            # Without this, SamplingParams.sampling_seed is accepted and then
            # SILENTLY IGNORED (sampling_batch_info.py:118-133) and sampling
            # falls back to the global torch RNG -- which makes a request's
            # output depend on who else is in its batch, exactly the
            # irreproducibility the transformers version had to document.
            # It also pins sampling_backend to pytorch, which is what fixes the
            # top_k/top_p composition semantics below.
            enable_deterministic_inference=True,
            # Pinned, not left to the fallback. Deterministic mode picks an
            # attention backend by GPU arch, and if the one it picks is not in
            # RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND
            # (= ascend/fa3/fa4/triton) it sets disable_radix_cache = True
            # behind a log line (server_args.py:7890-7895). Losing the radix
            # cache silently would cost ~M-fold on every probe while still
            # producing correct numbers, so it is pinned here and asserted
            # below. fa3 is the Hopper choice and is radix-supported.
            attention_backend=_attention_backend(model_path),
        )
        if not deterministic:
            args.pop("enable_deterministic_inference")
            args.pop("attention_backend")
        if context_length is not None:
            args["context_length"] = context_length
        if max_running_requests is not None:
            args["max_running_requests"] = max_running_requests
        args.update(kw)

        self.engine = Engine(**args)
        self.model_path = model_path
        _ENGINE_SINGLETON = self

        sa = getattr(self.engine, "server_args", None)
        if sa is not None and getattr(sa, "disable_radix_cache", False):
            self.close()
            raise SystemExit(
                "sglang disabled the radix cache. Every probe here is built on "
                "M requests sharing one prefix, so this would silently cost an "
                "M-fold slowdown. Check the attention backend / deterministic "
                "combination before rerunning.")
        if sa is not None and deterministic and not getattr(
                sa, "enable_deterministic_inference", False):
            self.close()
            raise SystemExit(
                "deterministic inference did not take; per-request "
                "sampling_seed would be silently ignored and the run would be "
                "irreproducible.")

    def close(self) -> None:
        try:
            self.engine.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ 0 --
    def warm_prefix(self, prefix) -> None:
        """Insert `prefix` into the radix tree before fanning out over it.

        Radix matching happens at admission against the tree AS IT IS AT THAT
        INSTANT. If M requests sharing a prefix are admitted in the same prefill
        batch while the tree is cold, each one prefills the whole prefix; the
        duplicate KV is freed on insert but the FLOPs are already spent. The
        default schedule policy is fcfs, so the cache-aware de-duplication in
        schedule_policy.py does not run either.

        One awaited scoring request first makes the other M hit the cache. This
        is the same warm-up pattern sglang uses internally.
        """
        if not prefix:
            return
        self.engine.generate(
            input_ids=[list(prefix)],
            sampling_params=[{"max_new_tokens": 0, "temperature": 1.0}],
        )

    # ------------------------------------------------------------------ 1 --
    def generate_ids(self, prompts_ids, policy, max_new_tokens,
                     stop_token_ids=None, ignore_eos=False, seed=None):
        """Batched free generation. Returns [(output_ids, finish_reason_type)].

        `seed` may be a scalar or one value per prompt. The per-prompt form is
        what makes a run reproducible independently of batch composition, and
        both backends accept it so callers never branch on which one they hold.

        `policy` is a CORPORA entry: {temp, top_p, top_k, ...}. top_k=0 in our
        config means DISABLED; sglang spells that -1, so it is translated here
        rather than at every call site.
        """
        seeds = (list(seed) if isinstance(seed, (list, tuple))
                 else [seed] * len(prompts_ids))
        if len(seeds) != len(prompts_ids):
            raise ValueError(f"{len(seeds)} seeds for {len(prompts_ids)} prompts")
        sps = [self._sampling_params(policy, max_new_tokens, stop_token_ids,
                                     ignore_eos, sd) for sd in seeds]
        out = self.engine.generate(
            input_ids=[list(p) for p in prompts_ids],
            sampling_params=sps,
        )
        if isinstance(out, dict):
            out = [out]
        res = []
        for o in out:
            fr = o["meta_info"].get("finish_reason") or {}
            res.append((list(o["output_ids"]), fr.get("type") if isinstance(fr, dict) else fr))
        return res

    @staticmethod
    def _sampling_params(policy, max_new_tokens, stop_token_ids=None,
                         ignore_eos=False, seed=None):
        sp = {
            "temperature": float(policy["temp"]),
            "top_p": float(policy["top_p"]),
            # config uses 0 for "disabled"; sglang uses -1. Passing 0 through
            # would be interpreted as a literal top-0 truncation.
            "top_k": int(policy["top_k"]) if policy["top_k"] else -1,
            "max_new_tokens": int(max_new_tokens),
            "ignore_eos": bool(ignore_eos),
        }
        if stop_token_ids:
            sp["stop_token_ids"] = list(stop_token_ids)
        if seed is not None:
            sp["sampling_seed"] = int(seed)
        return sp

    # ------------------------------------------------------------------ 2 --
    def teacher_forced_nll(self, seqs, start_lens):
        """-log p(seq[j] | seq[:j]) for every j >= start_len, per sequence.

        Raw head: these are INPUT logprobs, which sglang computes as a plain
        log_softmax of the unmodified logits.

        `start_lens[i]` is the first POSITION SCORED, so the returned list has
        len(seqs[i]) - start_lens[i] entries and entry n is the NLL of
        seqs[i][start_lens[i] + n].
        """
        if not seqs:
            return []
        # logprob_start_len = s makes the engine score input positions from s
        # onwards. Scoring position p requires the logits at p-1, hence s-1.
        starts = [max(0, s - 1) for s in start_lens]
        out = self.engine.generate(
            input_ids=[list(s) for s in seqs],
            sampling_params=[{"max_new_tokens": 0, "temperature": 1.0}] * len(seqs),
            return_logprob=True,
            logprob_start_len=starts,
        )
        if isinstance(out, dict):
            out = [out]
        res = []
        for seq, s, o in zip(seqs, start_lens, out):
            got = o["meta_info"]["input_token_logprobs"]
            res.append(_align_input_logprobs(got, seq, s))
        return res

    # ------------------------------------------------------------------ 2b -
    def teacher_forced_topk(self, seqs, start_lens, k_top):
        """Top-`k_top` next-token distribution at every scored position.

        CE only ever needs p(gt), a single number. TOTAL VARIATION needs the
        whole distribution, so this is the one primitive the TV floor T_k
        requires and the cross-entropy probes do not.

        Returns per sequence a list over scored positions, each a list of
        (token_id, logprob) of length <= k_top, sorted by descending logprob.
        The caller MUST report the residual mass 1 - sum(exp(logprob)): dropping
        the tail lowers every pairwise TV, which biases the floor DOWNWARD and
        therefore flatters the drafter being compared against it.
        """
        if not seqs:
            return []
        starts = [max(0, s - 1) for s in start_lens]
        out = self.engine.generate(
            input_ids=[list(s) for s in seqs],
            sampling_params=[{"max_new_tokens": 0, "temperature": 1.0}] * len(seqs),
            return_logprob=True,
            top_logprobs_num=int(k_top),
            logprob_start_len=starts,
        )
        if isinstance(out, dict):
            out = [out]
        res = []
        for seq, s, o in zip(seqs, start_lens, out):
            got = o["meta_info"].get("input_top_logprobs")
            if not got:
                raise RuntimeError(
                    "input_top_logprobs came back empty. This sglang build does "
                    "not populate the input-side top-logprob path, so the TV "
                    "floor cannot be measured on it.")
            want_n = len(seq) - s
            if len(got) != want_n + 1:
                raise RuntimeError(
                    f"input_top_logprobs has {len(got)} rows, expected "
                    f"{want_n+1} (same leading-None convention as §2).")
            res.append([_as_topk(g) for g in got[1:]])
        return res

    # ----------------------------------------------------------------- 2c -
    def teacher_forced_nll_topk(self, seqs, start_lens, k_top):
        """Both of the above from ONE engine call. Returns (nll, tops).

        The SNIS rung of the T ladder needs, on the same sequences, the NLL of
        the forced tokens (to build the importance weight) and the full
        next-token distribution at the last scored position (to build the
        barycentre). Issued separately those are two round trips over identical
        input, and at order 1 they are two round trips over the identical
        POSITION -- start_len is len(prefix)+k-1 and the distribution wanted is
        at len(seq)-1, which is the same index. Measured on the live probe, the
        two calls were 84% of wall clock with the accelerator reading 0%: the
        cost is round trips and logprob transfer, not arithmetic.

        sglang populates input_token_logprobs and input_top_logprobs from the
        same forward, so asking for both costs one extra field on the response
        rather than a second pass.
        """
        if not seqs:
            return [], []
        starts = [max(0, s - 1) for s in start_lens]
        out = self.engine.generate(
            input_ids=[list(s) for s in seqs],
            sampling_params=[{"max_new_tokens": 0, "temperature": 1.0}] * len(seqs),
            return_logprob=True,
            top_logprobs_num=int(k_top),
            logprob_start_len=starts,
        )
        if isinstance(out, dict):
            out = [out]
        nll, tops = [], []
        for seq, s, o in zip(seqs, start_lens, out):
            mi = o["meta_info"]
            nll.append(_align_input_logprobs(mi["input_token_logprobs"], seq, s))
            got = mi.get("input_top_logprobs")
            if not got:
                raise RuntimeError(
                    "input_top_logprobs came back empty. This sglang build does "
                    "not populate the input-side top-logprob path, so the TV "
                    "floor cannot be measured on it.")
            want_n = len(seq) - s
            if len(got) != want_n + 1:
                raise RuntimeError(
                    f"input_top_logprobs has {len(got)} rows, expected "
                    f"{want_n+1} (same leading-None convention as §2).")
            tops.append([_as_topk(g) for g in got[1:]])
        return nll, tops

    # ------------------------------------------------------------------ 3 --
    def teacher_forced_query(self, seqs, start_lens, query_ids):
        """logprobs of ARBITRARY ids at each scored position.

        Returns per sequence a list over scored positions p, each a dict
        {token_id: logprob} covering `query_ids[i]`. Used for CE_B: at the
        position that predicts slot k we want p(gt[k]) even though the path
        actually went somewhere else.
        """
        if not seqs:
            return []
        starts = [max(0, s - 1) for s in start_lens]
        out = self.engine.generate(
            input_ids=[list(s) for s in seqs],
            sampling_params=[{"max_new_tokens": 0, "temperature": 1.0}] * len(seqs),
            return_logprob=True,
            logprob_start_len=starts,
            token_ids_logprob=[list(q) for q in query_ids],
        )
        if isinstance(out, dict):
            out = [out]
        res = []
        for seq, s, q, o in zip(seqs, start_lens, query_ids, out):
            got = o["meta_info"].get("input_token_ids_logprobs")
            if not got:
                raise RuntimeError(
                    "input_token_ids_logprobs came back empty. This sglang build "
                    "does not populate the input-side token_ids_logprob path; "
                    "run the probes with --scoring-mode append.")
            res.append(_align_input_query(got, seq, s, q))
        return res


# ------------------------------------------------------------- alignment ----
# The two functions below are the ONLY place the input-logprob index convention
# is interpreted.
#
# The convention, from srt/managers/logprob_result_processor.py:38,50 -- with a
# request of length N and logprob_start_len = s, `input_token_logprobs` has
# exactly N - s entries and entry j is
#
#       (log p(input_ids[s+j] | input_ids[:s+j]), input_ids[s+j], text)
#
# EXCEPT entry 0, whose logprob is always None: the scheduler prepends a None
# and drops the final raw row, so the token at absolute position s never gets a
# logprob. That is why every caller here sends s = start_len - 1 and reads from
# index 1 -- sending s = start_len would silently return None for the first slot
# we care about.
#
# The token_id column is built independently of the value column, so entry j's
# id is an exact self-check on the alignment rather than a restatement of it.
# verify_backend.py additionally checks the whole thing numerically against
# transformers, so the convention is established twice, by different means.

def _rows_from(got, seq, start_len):
    """Drop the leading None row and assert the id column lines up."""
    want = list(seq[start_len:])
    if len(got) != len(want) + 1:
        raise RuntimeError(
            f"expected {len(want)+1} input logprob rows (N-s with s=start_len-1), "
            f"got {len(got)}. The logprob_start_len convention changed.")
    rows = got[1:]
    for n, r in enumerate(rows):
        tid = r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else None
        if tid is not None and int(tid) != want[n]:
            raise RuntimeError(
                f"input logprob misalignment at n={n}: row token id {tid} != "
                f"expected {want[n]}. Do not trust any number from this run.")
    return rows


def _align_input_logprobs(got, seq, start_len):
    """-log p(seq[j] | seq[:j]) for j >= start_len."""
    return [_nll(r[0]) for r in _rows_from(got, seq, start_len)]


def _align_input_query(got, seq, start_len, query_ids):
    """token_ids_logprob variant -> one {token_id: logprob} per scored position.

    Same leading-None convention: input_token_ids_logprobs[0] is a literal None
    rather than a list, so it is dropped before anything is parsed.
    """
    want_n = len(seq) - start_len
    if len(got) != want_n + 1:
        raise RuntimeError(
            f"input_token_ids_logprobs has {len(got)} rows, expected {want_n+1}")
    rows = [_as_id_map(g) for g in got[1:]]
    for r in rows:
        missing = [q for q in query_ids if q not in r]
        if missing:
            raise RuntimeError(f"requested ids missing from response: {missing[:4]}")
    return rows


def _as_id_map(entry):
    """Normalise one token_ids_logprob row to {token_id: logprob}.

    sglang returns either a list of (logprob, token_id, text) triples or a pair
    of parallel (values, indices) lists depending on the build, so both are
    accepted and anything else is rejected loudly.
    """
    if isinstance(entry, dict):
        return {int(k): float(v) for k, v in entry.items()}
    if (len(entry) == 2 and isinstance(entry[0], (list, tuple))
            and isinstance(entry[1], (list, tuple))
            and len(entry[0]) == len(entry[1])):
        return {int(i): float(v) for v, i in zip(entry[0], entry[1])}
    out = {}
    for item in entry:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out[int(item[1])] = float(item[0])
        else:
            raise RuntimeError(f"unrecognised token_ids_logprob row: {entry!r:.200}")
    return out


def _nll(lp):
    if lp is None:
        return float("nan")
    return -float(lp)


# ------------------------------------------------------------------ misc ----
def kv_bytes_per_token(cfg) -> int:
    # Gemma4's released config is a multimodal wrapper; the text tower's shape
    # lives in .text_config. See probe_rpre.text_cfg.
    cfg = getattr(cfg, "text_config", None) or cfg
    kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return cfg.num_hidden_layers * 2 * kv_heads * head_dim * 2


def weights_gib(model_path: str) -> float:
    """Size of the checkpoint's tensor files, in GiB."""
    import glob
    import os

    from transformers.utils import cached_file

    d = model_path
    if not os.path.isdir(d):
        try:                       # resolve a hub id to its snapshot directory
            d = os.path.dirname(cached_file(model_path, "config.json"))
        except Exception:
            return 8.5             # Qwen3-4B bf16; only used as a fallback
    tot = sum(os.path.getsize(f)
              for pat in ("*.safetensors", "*.bin")
              for f in glob.glob(os.path.join(d, pat)))
    return tot / (1 << 30) if tot else 8.5


def auto_mem_fraction(model_path: str, kv_gib: float = 12.0,
                      headroom_gib: float = 3.0, min_kv_gib: float = 2.0) -> float:
    """mem_fraction_static sized from what is ACTUALLY free right now.

    sglang's knob is NOT a fraction of total device memory. From
    mem_cache/kv_cache_configurator.py:1615-1637 the KV pool works out to

        kv = free_now - free_before * (1 - mem_fraction_static)

    i.e. mem_fraction_static is the fraction of the memory that was FREE AT
    STARTUP which sglang is allowed to keep; the rest is left as slack. So with
    F free and W of weights, claiming a KV pool of `kv` needs

        mem_fraction_static = (W + kv) / F

    which depends on the co-tenants' footprint at launch. A fixed value is
    therefore wrong in both directions on a shared box -- too small on a busy
    GPU (weights alone exceed the budget, hard abort) and needlessly greedy on
    an empty one. It is derived per launch and printed.
    """
    import torch

    free = torch.cuda.mem_get_info()[0] / (1 << 30)
    W = weights_gib(model_path)
    frac = min((W + kv_gib) / free, 1.0 - headroom_gib / free)
    kv = frac * free - W
    print(f"auto mem_fraction_static={frac:.3f}  "
          f"(free {free:.1f}GiB, weights {W:.1f}GiB -> KV pool ~{kv:.1f}GiB, "
          f"{free - frac*free:.1f}GiB left for co-tenants)", flush=True)
    if kv < min_kv_gib:
        raise SystemExit(
            f"only {free:.1f}GiB free and the weights need {W:.1f}GiB, leaving "
            f"{kv:.1f}GiB of KV -- too little. Pick an emptier GPU with "
            f"CUDA_VISIBLE_DEVICES.")
    return frac


def add_engine_args(ap):
    """Shared CLI surface, so every probe launches the engine identically."""
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="sglang mem_fraction_static; 0 = size from free VRAM")
    ap.add_argument("--context-length", type=int, default=None)
    ap.add_argument("--max-running", type=int, default=None,
                    help="cap concurrent requests if the KV pool is tight")
    return ap


def resolve_mem_fraction(v: float, model_path: str) -> float:
    return v if v and v > 0 else auto_mem_fraction(model_path)


def _as_topk(entry):
    """Normalise one input_top_logprobs row to [(token_id, logprob), ...].

    Same build-dependent shapes as _as_id_map: a list of (logprob, token_id,
    text) triples, or parallel (values, indices) lists. These two collide when
    a row happens to hold exactly two entries, so the parallel form is accepted
    ONLY when the second list is all integers and the first is not -- anything
    still ambiguous is rejected rather than guessed at, because a silently
    mis-parsed distribution would move every TV without looking wrong.
    """
    if entry is None:
        return []
    if isinstance(entry, dict):
        return sorted(((int(t), float(v)) for t, v in entry.items()),
                      key=lambda x: -x[1])
    if not isinstance(entry, (list, tuple)):
        raise RuntimeError(f"unparsable top-logprob entry: {type(entry)}")

    def _strict_ints(xs):
        # STRICT: token ids arrive as Python ints. Accepting integral floats
        # here misfires, because a logprob of exactly -2.0 is integral too and
        # would make a list of (logprob, id) pairs look like an id list.
        return bool(xs) and all(isinstance(x, int) and not isinstance(x, bool)
                                for x in xs)

    # parallel (values, indices): exactly two flat sequences of equal length,
    # the second all integral and the first not.
    if (len(entry) == 2
            and all(isinstance(x, (list, tuple)) for x in entry)
            and len(entry[0]) == len(entry[1])
            and not any(isinstance(v, (list, tuple)) for v in entry[0])
            and _strict_ints(entry[1]) and not _strict_ints(entry[0])):
        vals, ids = entry
        return sorted(((int(t), float(v)) for v, t in zip(vals, ids)),
                      key=lambda x: -x[1])

    # otherwise: rows of (logprob, token_id[, text])
    out = []
    for r in entry:
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            raise RuntimeError(f"unparsable top-logprob row: {r!r}")
        out.append((int(r[1]), float(r[0])))
    return sorted(out, key=lambda x: -x[1])

"""Corpus generation: C0 / C1 / C2 over a frozen prompt set.

Generates to NATURAL EOS. max_new_tokens is a safety valve only; anything that
hits it is flagged right_censored and its trailing block is unusable.

The same prompt IDs are reused across corpora and, later, across target models,
so every comparison is matched workload rather than re-sampled benchmarks.

Backend: sglang's in-process Engine. Besides being an order of magnitude faster
than HF `generate` at this shape (continuous batching, paged KV), it removes
three hand-rolled things the transformers version needed and could get wrong:

  * left padding. HF needed a padded batch and then an attention-mask-derived
    slice to recover each row's true prompt; sglang takes ragged input_ids.
  * stop handling. Qwen3 stops on a LIST ([151645, 151643]) and HF pads the tail
    with 151643, so the old code had to find the first stop itself and trim.
    sglang takes stop_token_ids and trims by default (no_stop_trim=False).
  * batch-dependent RNG. HF's sampler consumes one global generator, so a row's
    output depended on who else was in its batch -- hence the old `batch` field,
    which recorded the batch size purely so the RNG state was reconstructible.
    sglang has a PER-REQUEST `sampling_seed`, so each prompt is reproducible on
    its own. The seed actually used is recorded per sequence.

Usage:
  python -m specfloor.corpus --corpus C0 --domain gsm8k --out runs/C0/gsm8k.jsonl
"""

from __future__ import annotations

# backend must be imported before sglang: it sets SGLANG_RETURN_ORIGINAL_LOGPROB,
# which srt/layers/sampler.py latches at import time.
from specfloor import backend as B
from specfloor import backend_api as BA

import argparse
import hashlib
import json
import os
import pathlib

from transformers import AutoTokenizer, GenerationConfig

from specfloor import config as C

# Where the evaluation corpora live. Point SPECFLOOR_EVAL_ROOT at a directory
# holding the per-domain jsonl files; the default is a sibling of this package
# so a fresh clone works without configuration.
EVAL_ROOT = pathlib.Path(
    os.environ.get("SPECFLOOR_EVAL_ROOT",
                   pathlib.Path(__file__).resolve().parent.parent / "eval_datasets"))


def resolve_stop_token_ids(target_model, tokenizer):
    """Stop ids as a LIST, because Qwen3 has two.

    Inlined rather than imported so that measuring a floor needs nothing but
    transformers: a target model, a tokenizer, and this file. It reproduces the
    behaviour every serving stack implements -- prefer the generation config's
    eos, fall back to the tokenizer's, and always return a de-duplicated list,
    since a scalar eos silently truncates the stop set for models that declare
    several (Qwen3 stops on 151645 <|im_end|> and 151643 <|endoftext|>).
    """
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    out = []
    for token_id in eos_token_id:
        token_id = int(token_id)
        if token_id not in out:
            out.append(token_id)
    return out


class _GenCfgShim:
    """resolve_stop_token_ids() reads `.generation_config` off a loaded model.
    We only need the stop ids, so hand it the generation config directly rather
    than materialising 8GiB of weights on a shared GPU just to read two ints."""

    def __init__(self, path):
        self.generation_config = GenerationConfig.from_pretrained(path)


def resolve_stops(target: str, tok) -> list[int]:
    return resolve_stop_token_ids(_GenCfgShim(target), tok)


def _ids(enc) -> list[int]:
    """Normalise apply_chat_template(tokenize=True) to a flat list of ids.

    Depending on the transformers version this returns either a plain list[int]
    or a BatchEncoding. The BatchEncoding case is the dangerous one: it is
    dict-like, so len() is the NUMBER OF KEYS (2) and list() yields the key
    STRINGS -- which reaches sglang as a prompt of two tokens named
    'input_ids'/'attention_mask' rather than failing anywhere near the mistake.
    """
    if hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):     # batch-of-one
        enc = enc[0]
    if not enc or not isinstance(enc[0], int):
        raise TypeError(f"chat template did not yield token ids: {type(enc)}")
    return [int(x) for x in enc]


def prompt_id(domain: str, text: str) -> str:
    """Stable across models and runs -- this is what makes workloads matched."""
    h = hashlib.sha256(f"{domain}\x00{text}".encode()).hexdigest()[:16]
    return f"{domain}:{h}"


def load_prompts(domain: str, n: int) -> list[tuple[str, str]]:
    """Returns [(prompt_id, text)], deterministic order, deduplicated."""
    seen, out = set(), []
    with open(EVAL_ROOT / f"{domain}.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("turns") or d.get("prompt") or d.get("question")
            if isinstance(t, list):
                t = t[0]
            if not t or t in seen:
                continue
            seen.add(t)
            out.append((prompt_id(domain, t), t))
            if len(out) >= n:
                break
    return out


def seed_for(pid: str, seed: int) -> int:
    """Per-prompt seed: reproducible independently of batch composition, and
    stable if the prompt set is later extended or reordered."""
    h = hashlib.sha256(f"{pid}\x00{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=sorted(C.CORPORA))
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", type=int, default=C.PROMPTS_PER_DOMAIN)
    ap.add_argument("--batch", type=int, default=128,
                    help="requests submitted per generate() call; sglang "
                         "schedules within this, it does not affect sampling")
    ap.add_argument("--max-new-tokens", type=int, default=C.SAFETY_MAX_NEW_TOKENS)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--seed", type=int, default=C.SEED)
    B.add_engine_args(ap)
    BA.add_api_args(ap)
    args = ap.parse_args()

    policy = C.CORPORA[args.corpus]

    tok = AutoTokenizer.from_pretrained(args.target)
    stop_ids = resolve_stops(args.target, tok)
    assert stop_ids, "no stop tokens resolved -- refusing to guess"
    print(f"stop token ids: {stop_ids}", flush=True)

    prompts = load_prompts(args.domain, args.prompts)
    print(f"[{args.corpus}/{args.domain}] {len(prompts)} unique prompts", flush=True)
    if len(prompts) < args.prompts:
        print(f"  NOTE: dataset exhausted at {len(prompts)}; bootstrap cluster "
              f"stays the prompt either way", flush=True)

    enc = [
        _ids(tok.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=policy["thinking"],
        ))
        for _, t in prompts
    ]

    ctx_needed = max(len(e) for e in enc) + args.max_new_tokens
    print(f"longest prompt {max(len(e) for e in enc)} tok; "
          f"context needed {ctx_needed}", flush=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    censored = 0

    args.context_length = args.context_length or ctx_needed
    # need_logprobs=False: generation is the ONE stage that works at every
    # endpoint tier, so a chat-only API is permitted here and nowhere else.
    with BA.make_target(args, need_logprobs=False) as eng, out_path.open("w") as fout:
        for i0 in range(0, len(prompts), args.batch):
            chunk = prompts[i0: i0 + args.batch]
            ids = enc[i0: i0 + args.batch]
            seeds = [seed_for(pid, args.seed) for pid, _ in chunk]

            # per-request seeds, so a prompt's output does not depend on who
            # else is in its batch. Routed through generate_ids so the same code
            # serves the local engine and a remote endpoint.
            res = eng.generate_ids([list(x) for x in ids], policy,
                                   args.max_new_tokens, stop_ids, seed=seeds)

            for (pid, _), pids, sd, (gen, ftype) in zip(chunk, ids, seeds, res):
                hit_cap = (ftype == "length")
                # stop_token_ids are trimmed by sglang, but a stop token can
                # still appear if it is ALSO the natural continuation; drop any
                # trailing stop token defensively so response_ids never ends on
                # one (the anchor population assumes it does not).
                while gen and gen[-1] in stop_ids:
                    gen.pop()
                censored += hit_cap
                fout.write(json.dumps({
                    "prompt_id": pid,
                    "corpus": args.corpus,
                    "domain": args.domain,
                    "prompt_ids": [int(x) for x in pids],
                    "prompt_len": len(pids),
                    "response_ids": gen,
                    "response_len": len(gen),
                    "right_censored": bool(hit_cap),
                    "sampling_seed": sd,   # per-request; batch-independent
                    "seed": args.seed,
                }) + "\n")

            done = i0 + len(chunk)
            print(f"  {done}/{len(prompts)}  censored={censored}", flush=True)

    rate = censored / max(1, len(prompts))
    print(f"== {args.corpus}/{args.domain}: {len(prompts)} sequences, "
          f"censor rate {rate:.4%}", flush=True)
    if rate > C.MAX_CENSOR_RATE:
        # EXIT NONZERO. Right-censoring truncates the longest responses, which
        # are exactly the ones that carry the deep context buckets and the late
        # relative-position stratum, so a corpus over the gate changes both the
        # block weights and the stratification of everything downstream. This
        # used to print and exit 0, and a run that exceeded the gate flowed
        # into published numbers because the next stage only saw success.
        raise SystemExit(
            f"!! censor rate {rate:.4%} exceeds {C.MAX_CENSOR_RATE:.2%}. "
            f"Raise --max-new-tokens and regenerate; this corpus must not be "
            f"consumed by the anchor or probe stages.")


if __name__ == "__main__":
    main()

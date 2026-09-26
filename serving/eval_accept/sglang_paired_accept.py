"""Record PER-REQUEST serving accept length, so arms can be compared as paired samples.

WHY PER-REQUEST.  `sglang_multids_accept.py` prints a mean per dataset.  Two arms
measured that way are two independent samples, and the between-prompt variance (a short
factual answer and a long derivation have very different accept lengths) sits inside the
error bar of both.  Pairing removes it: the same prompt run through two arms differs only
by the drafter, so the paired delta has the prompt effect subtracted out exactly.  On the
held-out offline ruler the same change was worth roughly a 2.5x tighter interval.

WHAT MAKES THE PAIRING EXACT AT TEMPERATURE 0.  Lossless speculative decoding emits the
token the TARGET would have emitted, so at temp 0 the output text does not depend on the
drafter at all -- only the number of verify steps does.  This records a hash of the output
so that invariant is CHECKED rather than assumed: if two arms disagree on the text at
temp 0, one of them is not lossless, and that is a far more serious finding than any
accept-length difference.  At temp > 0 the hash will differ legitimately and the
comparison degrades to "same prompts, different draws" -- still paired on the prompt, but
without the losslessness check.

Accept length is BATCH-INVARIANT (completion_tokens / spec_verify_ct is unaffected by how
requests are batched), so concurrency is free here.  Speedup is NOT -- measure it
separately at bs=1, never from this harness.

Usage:
  EVAL_DATASETS=<DeepSpec>/eval_datasets \
  python sglang_paired_accept.py <server_url> <out.json> [temperature] [concurrency] [cap]

Environment: EVAL_DATASETS (required), SEED, OFFICIAL_SUBSET=1, MAXTOK (2048),
LABEL, TOKENIZER (Qwen/Qwen3-4B).
"""
import concurrent.futures as cf
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer

SERVER = sys.argv[1]
OUT = sys.argv[2]
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 32
CAP = int(sys.argv[5]) if len(sys.argv) > 5 else 100
MAXTOK = int(os.environ.get("MAXTOK", "2048"))
SEED = os.environ.get("SEED")
# opt-in: replicate DeepSpec's seeded-shuffle subsetting instead of a prefix
OFFICIAL_SUBSET = os.environ.get("OFFICIAL_SUBSET", "") == "1"
LABEL = os.environ.get("LABEL", os.path.basename(OUT).removesuffix(".json"))
EVAL_DATASETS = os.environ["EVAL_DATASETS"]

# Same list and caps as sglang_multids_accept.py, so paired numbers and the historical
# unpaired ones are drawn from the same population.
DATASETS = [("gsm8k", 500), ("math500", 500), ("aime25", 30), ("humaneval", 164),
            ("mbpp", 256), ("livecodebench", 500), ("mt-bench", 80), ("alpaca", 500),
            ("arena-hard-v2", 500)]

tok = AutoTokenizer.from_pretrained(os.environ.get("TOKENIZER", "Qwen/Qwen3-4B"))


def read_prompts(name, cap):
    """Prompts for one task.

    Default: the first `cap` rows, which is what every recorded run so far used.

    OFFICIAL_SUBSET=1 instead replicates DeepSpec's own selection
    (base_evaluator.py:544-548): read the whole file, and only if it is LARGER than the
    cap, shuffle it with random.Random(seed) and take the first `cap`.  Shuffling the
    extracted strings is equivalent to shuffling the rows -- random.shuffle applies a
    permutation of positions and never looks at the elements -- so this reproduces their
    subset exactly given the same seed.  Files at or below the cap keep file order, so
    humaneval/mt-bench/aime25/math500 at their official counts are untouched either way.

    The two modes select DIFFERENT prompts, so numbers from one are not paired with the
    other; that is why this is opt-in rather than a fix applied in place.
    """
    out = []
    with open(os.path.join(EVAL_DATASETS, f"{name}.jsonl")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = d.get("turns") or d.get("prompt") or d.get("question")
            if isinstance(t, list):
                t = t[0]
            if t:
                out.append(t)
            if not OFFICIAL_SUBSET and len(out) >= cap:
                break
    if OFFICIAL_SUBSET and len(out) > cap:
        rng = random.Random(int(SEED if SEED is not None else 980406))
        rng.shuffle(out)
        out = out[:cap]
    return out


def fmt(question):
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def run_one(text):
    params = {"temperature": TEMP, "max_new_tokens": MAXTOK}
    if SEED is not None:
        params["sampling_seed"] = int(SEED)
    body = json.dumps({"text": text, "sampling_params": params}).encode()
    req = urllib.request.Request(
        SERVER + "/generate", data=body, headers={"Content-Type": "application/json"}
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    info = resp.get("meta_info", {})
    completion = int(info.get("completion_tokens", 0))
    verify_ct = info.get("spec_verify_ct", None)
    return {
        "completion_tokens": completion,
        "spec_verify_ct": verify_ct,
        # tau, the EAGLE convention: tokens per verify step, bonus included.
        "accept": (completion / verify_ct) if verify_ct else None,
        "output_sha1": hashlib.sha1(resp.get("text", "").encode()).hexdigest()[:16],
    }


def guarded(idx_text):
    idx, text = idx_text
    try:
        return idx, run_one(text)
    except Exception as exc:  # a dead request must not silently shrink the sample
        return idx, {"error": f"{type(exc).__name__}: {exc}"}


try:
    run_one(fmt("What is 2+2?"))
except Exception as exc:
    print("warmup err", exc)

records = []
print(f"server={SERVER} label={LABEL} temp={TEMP} conc={CONC} cap={CAP} maxtok={MAXTOK}")
print(f"{'dataset':<16}{'n':>5}{'accept':>9}{'err':>5}{'sec':>8}")
for name, cap in DATASETS:
    prompts = [fmt(p) for p in read_prompts(name, min(cap, CAP))]
    start = time.time()
    got = [None] * len(prompts)
    with cf.ThreadPoolExecutor(max_workers=CONC) as pool:
        for idx, rec in pool.map(guarded, list(enumerate(prompts))):
            got[idx] = rec
    ok = [r["accept"] for r in got if r.get("accept") is not None]
    errs = sum(1 for r in got if "error" in r)
    for idx, rec in enumerate(got):
        # The prompt HASH, not the prompt: it is the join key across arms, and storing
        # it instead of the text keeps a result file small enough to keep forever.
        rec.update(dataset=name, idx=idx,
                   prompt_sha1=hashlib.sha1(prompts[idx].encode()).hexdigest()[:16])
        records.append(rec)
    mean = sum(ok) / max(1, len(ok))
    print(f"{name:<16}{len(ok):>5}{mean:>9.3f}{errs:>5}{time.time() - start:>8.1f}",
          flush=True)

payload = {
    "label": LABEL, "server": SERVER, "temperature": TEMP, "max_new_tokens": MAXTOK,
    "seed": SEED, "cap": CAP, "official_subset": OFFICIAL_SUBSET,
    "records": records,
}
with open(OUT, "w") as fh:
    json.dump(payload, fh)
ok_all = [r["accept"] for r in records if r.get("accept") is not None]
print(f"{'MACRO(micro)':<16}{len(ok_all):>5}{sum(ok_all) / max(1, len(ok_all)):>9.3f}")
print(f"wrote {OUT}")

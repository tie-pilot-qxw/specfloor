"""Build the 3,030 serving requests from DeepSpec's evaluation sets.

    python prepare_requests.py --eval-datasets <DeepSpec>/eval_datasets --out prepared_requests.jsonl

For each of the nine tasks: read `<task>.jsonl`, take the first turn of every
row, and, when the file has more rows than the task's count, shuffle it with
random.Random(980406) and keep the first `count` (DeepSpec's own selection rule).
Each prompt is formatted with the Qwen3-4B chat template, thinking disabled, and
tokenized once. The script checks every input file and the output against the
SHA-256 values the archived sweep recorded and refuses to write a different set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from common import DATASETS, PREPARED_SHA256, ROWS, SEED, TARGET

DATASET_SHA256 = {
    "gsm8k": "63330a20a17f416fdfca978ebcd5124f8da5546affb3a079b65a6e8daf42b41f",
    "math500": "4f530d9d3126dca41c12a96e78ec0dd460b7202052ba86101c4da159cea33aac",
    "aime25": "b8835996839caa1d982c5bb9ecaeca636424f0b8ed9f33a40b084feb1c1766d0",
    "humaneval": "7aaed6a3987007ecee4851ee4572fde214aa2a36ae83fa21be9c5f443aa71675",
    "mbpp": "019b981f7122e39ce58866cf93f4206511c72532cb537bbaf298753a0f128ba7",
    "livecodebench": "40ed536492f331ac27b3b68506072366ca1d3a304b92c31d822574e778c87318",
    "mt-bench": "df05defbcea350dff39f7d2996a7cba8a029e6e06069dfcdab11ceb998d743b0",
    "alpaca": "c1f1623ac08e4c4f024604bb2689a024963eb7eb9dbbc29e57dda5a11d87e07b",
    "arena-hard-v2": "e5fcce94ffb1a2ef2082b5c63ac36608f2b7faf00da13ecb94ce2adc649b7b91",
}


def prepare(eval_root, tokenizer):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = []
    for name, cap in DATASETS:
        source = eval_root / (name + ".jsonl")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != DATASET_SHA256[name]:
            raise ValueError(f"{source} differs from the archived dataset ({digest})")
        texts = []
        for line in source.read_text().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            texts.append(d["turns"][0] if isinstance(d["turns"], list) else d["turns"])
        if len(texts) > cap:
            random.Random(SEED).shuffle(texts)
            texts = texts[:cap]
        if len(texts) != cap:
            raise ValueError(f"Wrong count for {name}")
        for i, text in enumerate(texts):
            formatted = tok.apply_chat_template([{"role": "user", "content": text}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(formatted)
            rows.append({"dataset": name, "idx": i, "input_ids": ids, "text": formatted,
                         "prompt_sha1": hashlib.sha1(text.encode()).hexdigest(),
                         "input_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-datasets", type=Path, required=True,
                    help="DeepSpec's eval_datasets/ directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", default=TARGET)
    args = ap.parse_args()
    rows = prepare(args.eval_datasets, args.tokenizer)
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    digest = hashlib.sha256(data).hexdigest()
    if len(rows) != ROWS or digest != PREPARED_SHA256:
        raise SystemExit(f"Built {len(rows)} rows with SHA-256 {digest}, not the archived "
                         f"{PREPARED_SHA256}; check the tokenizer revision.")
    args.out.write_bytes(data)
    print(f"wrote {args.out}: {len(rows)} rows, sha256 {digest}")


if __name__ == "__main__":
    main()

"""Convert validated conversation rollouts to online-trainer token records.

This uses the same ``preprocess_record`` function as the official target-cache
builder.  Every assistant span contributes to ``loss_mask``; user/system spans
do not.  Prefix truncation and the minimum supervised-token filter therefore
match ``ConversationCollator`` without materializing a multi-terabyte hidden
cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from pathlib import Path

from transformers import AutoTokenizer

from deepspec.data.parser import preprocess_record

try:
    from .jsonl_resume import iter_resume_records
except ImportError:  # Direct execution: python scripts/data/tokenize_official_rollouts.py
    from jsonl_resume import iter_resume_records


FORMAT = "deepspec-official-multiturn-online-v2"
_WORKER_TOKENIZER = None
_WORKER_CHAT_TEMPLATE = None
_WORKER_MAX_LENGTH = None
_WORKER_MIN_LOSS_TOKENS = None


def tokenize_record(
    record: dict,
    *,
    tokenizer,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
) -> tuple[bool, dict]:
    """Tokenize one rollout without changing the official record semantics."""
    source_id = int(record["id"])
    processed = preprocess_record(
        record=record,
        tokenizer=tokenizer,
        chat_template=chat_template,
        max_length=max_length,
    )
    input_ids = processed["input_ids"].tolist()
    loss_mask = [int(value != 0) for value in processed["loss_mask"].tolist()]
    loss_tokens = sum(loss_mask)
    common = {
        "source_id": source_id,
        "source_user_turns": sum(
            message.get("role") == "user"
            for message in record["conversations"]
        ),
    }
    if loss_tokens < min_loss_tokens:
        return False, {
            **common,
            "reason": "too_few_loss_tokens_after_prefix_truncation",
            "loss_tokens": loss_tokens,
            "sequence_length": len(input_ids),
        }
    return True, {
        **common,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }


def init_tokenizer_worker(
    target: str,
    chat_template: str,
    max_length: int,
    min_loss_tokens: int,
) -> None:
    global _WORKER_TOKENIZER
    global _WORKER_CHAT_TEMPLATE
    global _WORKER_MAX_LENGTH
    global _WORKER_MIN_LOSS_TOKENS
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(target)
    _WORKER_CHAT_TEMPLATE = chat_template
    _WORKER_MAX_LENGTH = max_length
    _WORKER_MIN_LOSS_TOKENS = min_loss_tokens


def tokenize_record_in_worker(record: dict) -> tuple[bool, dict]:
    assert _WORKER_TOKENIZER is not None
    return tokenize_record(
        record,
        tokenizer=_WORKER_TOKENIZER,
        chat_template=_WORKER_CHAT_TEMPLATE,
        max_length=_WORKER_MAX_LENGTH,
        min_loss_tokens=_WORKER_MIN_LOSS_TOKENS,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_done_source_ids(paths: list[Path]) -> set[int]:
    done = set()
    for path in paths:
        for line_number, record in iter_resume_records(path):
            source_id = int(record["source_id"])
            if source_id in done:
                raise ValueError(
                    f"duplicate source_id={source_id} across token/rejection "
                    f"outputs (at {path}:{line_number})"
                )
            done.add(source_id)
    return done


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--chat-template", choices=("qwen", "gemma4"), required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--expected-total", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "CPU tokenization processes. Results are consumed in input order, so "
            "changing this does not change the generated dataset."
        ),
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=1024,
        help="Maximum records submitted to the worker pool at once.",
    )
    args = parser.parse_args()
    if args.max_length != 4096:
        parser.error("formal online data requires --max-length 4096")
    if args.min_loss_tokens < 1:
        parser.error("--min-loss-tokens must be positive")
    if args.expected_total is not None and args.expected_total < 1:
        parser.error("--expected-total must be positive")
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    if args.worker_batch_size < 1:
        parser.error("--worker-batch-size must be positive")
    return args


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    rejected_path = Path(args.rejected_output).resolve()
    metadata_path = output_path.with_suffix(f"{output_path.suffix}.meta.json")
    completion_path = output_path.with_suffix(f"{output_path.suffix}.complete.json")

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if len({input_path, output_path, rejected_path}) != 3:
        raise ValueError("input, output and rejected-output must be different files")

    metadata = {
        "format": FORMAT,
        "source": str(input_path),
        "source_sha256": sha256_file(input_path),
        "target": args.target,
        "chat_template": args.chat_template,
        "max_length": args.max_length,
        "min_loss_tokens": args.min_loss_tokens,
    }
    existing_paths = [
        path
        for path in (output_path, rejected_path, metadata_path, completion_path)
        if path.exists()
    ]
    if existing_paths and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing_paths}")
    if args.resume:
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"cannot safely resume without conversion metadata: {metadata_path}"
            )
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata != metadata:
            raise ValueError(
                f"resume metadata mismatch for {metadata_path}: "
                f"{existing_metadata!r} != {metadata!r}"
            )
        # A resumed conversion is not complete again until the full source has
        # been rescanned and exact coverage re-established below.
        completion_path.unlink(missing_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        atomic_write_json(metadata_path, metadata)

    accepted_before = (
        load_done_source_ids([output_path]) if args.resume else set()
    )
    rejected_before = (
        load_done_source_ids([rejected_path]) if args.resume else set()
    )
    overlap = accepted_before & rejected_before
    if overlap:
        raise ValueError(
            f"source ids appear in both token and rejection outputs: "
            f"{sorted(overlap)[:20]}"
        )
    done = accepted_before | rejected_before
    tokenizer = (
        AutoTokenizer.from_pretrained(args.target) if args.num_workers == 1 else None
    )
    accepted = rejected = skipped = rows = 0
    seen_source_ids = set()
    output_mode = "a" if args.resume else "w"
    if args.num_workers == 1:
        process_one = partial(
            tokenize_record,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            min_loss_tokens=args.min_loss_tokens,
        )
        executor_context = nullcontext(None)
    else:
        process_one = tokenize_record_in_worker
        executor_context = ProcessPoolExecutor(
            max_workers=args.num_workers,
            initializer=init_tokenizer_worker,
            initargs=(
                args.target,
                args.chat_template,
                args.max_length,
                args.min_loss_tokens,
            ),
        )

    with (
        input_path.open("r", encoding="utf-8") as input_handle,
        output_path.open(output_mode, encoding="utf-8") as output_handle,
        rejected_path.open(output_mode, encoding="utf-8") as rejected_handle,
        executor_context as executor,
    ):
        pending_records = []

        def process_pending() -> None:
            nonlocal accepted, rejected
            if not pending_records:
                return
            results = (
                map(process_one, pending_records)
                if args.num_workers == 1
                else executor.map(process_one, pending_records, chunksize=8)
            )
            for is_accepted, payload in results:
                handle = output_handle if is_accepted else rejected_handle
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                if is_accepted:
                    accepted += 1
                else:
                    rejected += 1

                processed_count = accepted + rejected
                if processed_count % 1000 == 0:
                    output_handle.flush()
                    rejected_handle.flush()
                    print(
                        f"[tokenize-official] accepted={accepted} rejected={rejected} "
                        f"skipped={skipped} processed={processed_count}",
                        flush=True,
                    )
            pending_records.clear()

        for line_number, line in enumerate(input_handle, 1):
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            source_id = int(record["id"])
            if source_id in seen_source_ids:
                raise ValueError(f"duplicate source id={source_id} at input line {line_number}")
            seen_source_ids.add(source_id)
            if source_id in done:
                skipped += 1
                continue
            pending_records.append(record)
            if len(pending_records) >= args.worker_batch_size:
                process_pending()
        process_pending()

    unexpected_done = done - seen_source_ids
    if unexpected_done:
        raise ValueError(
            f"resume outputs contain {len(unexpected_done)} source ids absent from "
            f"input, examples={sorted(unexpected_done)[:20]}"
        )
    if args.expected_total is not None and rows != args.expected_total:
        raise ValueError(
            f"input coverage mismatch: expected {args.expected_total}, found {rows}"
        )
    if accepted + rejected + skipped != rows:
        raise AssertionError("tokenization accounting mismatch")
    completion = {
        **metadata,
        "input_rows": rows,
        "accepted_rows": len(accepted_before) + accepted,
        "rejected_rows": len(rejected_before) + rejected,
    }
    if completion["accepted_rows"] + completion["rejected_rows"] != rows:
        raise AssertionError("completion coverage mismatch")
    atomic_write_json(completion_path, completion)
    print(
        f"[tokenize-official] input={input_path} rows={rows} accepted={accepted} "
        f"rejected={rejected} skipped={skipped} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

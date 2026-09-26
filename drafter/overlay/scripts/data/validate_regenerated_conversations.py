"""Validate official regenerated shards against their source conversation shards."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def user_turn_fingerprint(conversations):
    users = [
        message.get("content")
        for message in conversations
        if message.get("role") == "user"
    ]
    if not users or any(not isinstance(content, str) or not content for content in users):
        raise ValueError("missing or empty user turn")
    payload = json.dumps(users, ensure_ascii=False, separators=(",", ":")).encode()
    return len(users), hashlib.blake2b(payload, digest_size=16).digest()


def load_source_contract(paths):
    contract = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                source_id = int(record["id"])
                if source_id in contract:
                    raise ValueError(f"duplicate source id={source_id} at {path}:{line_number}")
                contract[source_id] = user_turn_fingerprint(record["conversations"])
    return contract


def validate_regenerated_record(record, expected):
    if record.get("status") != "success":
        raise ValueError("record status is not success")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("missing conversations")
    body = conversations[1:] if conversations[0].get("role") == "system" else conversations
    if len(body) % 2:
        raise ValueError("regenerated body is not user/assistant pairs")
    for offset in range(0, len(body), 2):
        user = body[offset]
        assistant = body[offset + 1]
        if user.get("role") != "user" or assistant.get("role") != "assistant":
            raise ValueError("regenerated roles do not alternate user/assistant")
        if not isinstance(assistant.get("content"), str) or not assistant["content"]:
            raise ValueError("empty regenerated assistant turn")
    if user_turn_fingerprint(conversations) != expected:
        raise ValueError("source user turns changed during regeneration")


def merge_jsonl(paths, output_path):
    output_path = os.path.abspath(output_path)
    if output_path in paths:
        raise ValueError("--merge-out must not overwrite an input shard")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "wb") as output_handle:
        for path in paths:
            with open(path, "rb") as input_handle:
                for line in input_handle:
                    if line.strip():
                        output_handle.write(line)
    os.replace(temporary_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_inputs", nargs="+")
    parser.add_argument("--source-inputs", nargs="+", required=True)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--merge-out")
    args = parser.parse_args()
    source_paths = expand_inputs(args.source_inputs)
    generated_paths = expand_inputs(args.generated_inputs)
    for path in source_paths + generated_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    contract = load_source_contract(source_paths)
    if len(contract) != args.expected_total:
        raise ValueError(
            f"source coverage mismatch: expected {args.expected_total}, found {len(contract)}"
        )
    seen = set()
    errors = []
    rows = 0
    for path in generated_paths:
        digest = hashlib.sha256()
        shard_rows = 0
        with open(path, "rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                rows += 1
                shard_rows += 1
                try:
                    record = json.loads(raw_line)
                    source_id = int(record["id"])
                    if source_id not in contract:
                        raise ValueError(f"unexpected source id={source_id}")
                    if source_id in seen:
                        raise ValueError(f"duplicate generated id={source_id}")
                    validate_regenerated_record(record, contract[source_id])
                    seen.add(source_id)
                except Exception as exc:
                    if len(errors) < 50:
                        errors.append(f"{path}:{line_number}: {exc}")
        print(
            json.dumps(
                {"path": path, "rows": shard_rows, "sha256": digest.hexdigest()},
                ensure_ascii=False,
            )
        )
    missing = set(contract) - seen
    print(
        json.dumps(
            {
                "source_rows": len(contract),
                "generated_rows": rows,
                "missing": len(missing),
                "missing_examples": sorted(missing)[:20],
                "errors": len(errors),
            }
        )
    )
    if errors or missing or rows != args.expected_total:
        for error in errors:
            print(f"[validate-regenerated] ERROR {error}")
        raise SystemExit(1)
    if args.merge_out:
        merge_jsonl(generated_paths, args.merge_out)


if __name__ == "__main__":
    main()

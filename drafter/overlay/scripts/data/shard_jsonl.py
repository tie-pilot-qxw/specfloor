"""Split an official train JSONL into deterministic contiguous worker shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def compute_shard_bounds(total: int, shard_index: int, num_shards: int):
    base, remainder = divmod(total, num_shards)
    start = shard_index * base + min(shard_index, remainder)
    count = base + int(shard_index < remainder)
    return start, count


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--expected-total", required=True, type=int)
    args = parser.parse_args()
    if args.num_shards <= 0 or args.expected_total <= 0:
        parser.error("--num-shards and --expected-total must be positive")
    return args


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    shard_paths = [
        args.output_dir / f"shard_{index:05d}_of_{args.num_shards:05d}.jsonl"
        for index in range(args.num_shards)
    ]
    existing = [path for path in [manifest_path, *shard_paths] if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing shard artifacts: "
            + ", ".join(str(path) for path in existing[:5])
        )

    temporary_paths = [path.with_suffix(".jsonl.tmp") for path in shard_paths]
    handles = [path.open("wb") for path in temporary_paths]
    source_digest = hashlib.sha256()
    shard_digests = [hashlib.sha256() for _ in shard_paths]
    shard_counts = [0] * args.num_shards
    boundaries = [
        compute_shard_bounds(args.expected_total, index, args.num_shards)
        for index in range(args.num_shards)
    ]
    shard_index = 0
    try:
        with args.input.open("rb") as input_handle:
            for row_index, line in enumerate(input_handle):
                if row_index >= args.expected_total:
                    raise ValueError(
                        f"input has more than expected {args.expected_total} rows"
                    )
                while (
                    shard_index + 1 < args.num_shards
                    and row_index >= boundaries[shard_index][0] + boundaries[shard_index][1]
                ):
                    shard_index += 1
                record = json.loads(line)
                if "id" not in record:
                    raise ValueError(f"input row {row_index} is missing id")
                source_digest.update(line)
                handles[shard_index].write(line)
                shard_digests[shard_index].update(line)
                shard_counts[shard_index] += 1
        total = sum(shard_counts)
        if total != args.expected_total:
            raise ValueError(
                f"input row count mismatch: expected {args.expected_total}, found {total}"
            )
    except Exception:
        for handle in handles:
            handle.close()
        raise
    else:
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    for temporary_path, final_path in zip(temporary_paths, shard_paths):
        os.replace(temporary_path, final_path)
    manifest = {
        "format": "deepspec-official-jsonl-shards-v1",
        "source_path": str(args.input.resolve()),
        "source_sha256": source_digest.hexdigest(),
        "expected_total": args.expected_total,
        "num_shards": args.num_shards,
        "shards": [
            {
                "index": index,
                "path": path.name,
                "start": boundaries[index][0],
                "rows": shard_counts[index],
                "sha256": shard_digests[index].hexdigest(),
            }
            for index, path in enumerate(shard_paths)
        ],
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

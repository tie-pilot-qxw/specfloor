"""Verify a worker JSONL shard against the coordinator's shard manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


MANIFEST_FORMAT = "deepspec-official-jsonl-shards-v1"


def validate_sha256(value, label):
    value = str(value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{label} is missing or malformed")
    return value.lower()


def sha256_and_rows(path: Path):
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            if line.strip():
                rows += 1
    return digest.hexdigest(), rows


def verify_shard(
    *,
    manifest_path: Path,
    shard_path: Path,
    shard_index: int,
    num_shards: int,
    expected_total: int,
    expected_source_sha256: str | None = None,
):
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing shard manifest: {manifest_path}")
    if not shard_path.is_file():
        raise FileNotFoundError(f"missing shard file: {shard_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"unexpected manifest format: {manifest.get('format')!r}")
    if int(manifest.get("expected_total", -1)) != expected_total:
        raise ValueError(
            "manifest expected_total mismatch: "
            f"{manifest.get('expected_total')} != {expected_total}"
        )
    if int(manifest.get("num_shards", -1)) != num_shards:
        raise ValueError(
            f"manifest num_shards mismatch: {manifest.get('num_shards')} != {num_shards}"
        )
    source_sha256 = validate_sha256(
        manifest.get("source_sha256", ""), "manifest source_sha256"
    )
    if expected_source_sha256 is not None and source_sha256 != validate_sha256(
        expected_source_sha256, "expected source SHA256"
    ):
        raise ValueError(
            f"manifest source SHA256 mismatch: {source_sha256} != {expected_source_sha256}"
        )

    entries = manifest.get("shards")
    if not isinstance(entries, list) or len(entries) != num_shards:
        raise ValueError("manifest shard entry count does not match num_shards")
    by_index = {}
    expected_start = 0
    for entry in entries:
        index = int(entry["index"])
        if index in by_index or not 0 <= index < num_shards:
            raise ValueError(f"invalid or duplicate manifest shard index: {index}")
        rows = int(entry["rows"])
        start = int(entry["start"])
        if rows < 0 or start != expected_start:
            raise ValueError(
                f"manifest shard ranges are not contiguous at index={index}: "
                f"start={start}, expected={expected_start}, rows={rows}"
            )
        entry["sha256"] = validate_sha256(
            entry.get("sha256", ""), f"manifest shard {index} SHA256"
        )
        by_index[index] = entry
        expected_start += rows
    if expected_start != expected_total:
        raise ValueError(
            f"manifest shard rows sum to {expected_start}, expected {expected_total}"
        )

    entry = by_index[shard_index]
    if os.path.basename(str(entry["path"])) != shard_path.name:
        raise ValueError(
            f"shard filename mismatch: {shard_path.name} != {entry['path']}"
        )
    actual_sha256, actual_rows = sha256_and_rows(shard_path)
    if actual_rows != int(entry["rows"]):
        raise ValueError(
            f"shard row-count mismatch: {actual_rows} != {entry['rows']}"
        )
    if actual_sha256 != entry["sha256"]:
        raise ValueError(
            f"shard SHA256 mismatch: {actual_sha256} != {entry['sha256']}"
        )
    return {
        "manifest": str(manifest_path.resolve()),
        "source_sha256": source_sha256,
        "shard": str(shard_path.resolve()),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "rows": actual_rows,
        "sha256": actual_sha256,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--expected-total", required=True, type=int)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.expected_total <= 0:
        parser.error("--expected-total must be positive")
    report = verify_shard(
        manifest_path=args.manifest,
        shard_path=args.shard,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        expected_total=args.expected_total,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""Helpers for safely resuming append-only JSONL outputs."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _has_non_whitespace_remaining(handle) -> bool:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        if chunk.strip():
            return True
    return False


def iter_resume_records(path: str | Path) -> Iterator[tuple[int, Any]]:
    """Yield JSONL records, repairing only a malformed final nonempty line.

    Resume outputs are append-only, so a process killed during ``write`` may
    leave the final JSON value incomplete. Such a tail is removed so the row
    can be regenerated cleanly. A malformed line with any later non-whitespace
    content is not a crash tail and its original decode error is raised.
    """

    path_string = os.fspath(path)
    if not os.path.exists(path_string):
        return

    with open(path_string, "r+b") as handle:
        last_valid_line_missing_newline = False
        line_number = 0
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                last_valid_line_missing_newline = False
                continue

            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if _has_non_whitespace_remaining(handle):
                    raise
                handle.seek(line_start)
                handle.truncate()
                handle.flush()
                os.fsync(handle.fileno())
                print(
                    f"[resume] removed malformed final JSONL line "
                    f"{path_string}:{line_number}",
                    flush=True,
                )
                return

            last_valid_line_missing_newline = not line.endswith(b"\n")
            yield line_number, record

        # A fully written JSON value can still be missing only its terminating
        # newline. Preserve that completed row and make the next append safe.
        if last_valid_line_missing_newline:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


"""Guard the one alignment that a rescoring probe can silently get wrong.

`teacher_forced_nll_topk` returns one row per token of the sequence, and row i
is the distribution that PREDICTED token i -- computed from seq[:i]. So
`tops[-1]` conditions on everything before the last token, not on the last
token. A probe that wants the distribution AFTER a revealed suffix must put one
more token on the sequence to create that row.

probe_tk once did not, and its m>=1 numbers were the weighted spread of
p(. | X, s) instead of the floor of p(. | X, s, z*) -- an off-by-one slot, with
the importance weights themselves correct, so nothing downstream looked wrong.
probe_rm has always built the sequence correctly and is checked here too so the
two cannot drift apart again.

    python -m pytest tests/test_slot_alignment.py
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent / "specfloor"


def body(path, name):
    """The source of one function, from its def to the next same-level def."""
    src = (ROOT / path).read_text()
    i = src.index(f"def {name}")
    indent = len(src[:i].split("\n")[-1])
    rest = src[i:]
    nxt = re.search(rf"\n{' ' * indent}def ", rest)
    return rest[: nxt.start()] if nxt else rest


def test_backend_contract_is_what_the_probes_assume():
    src = body("backend.py", "teacher_forced_nll_topk")
    # logprob_start_len is pulled back one so the row for position s exists,
    # and the leading None of that convention is dropped again.
    assert "starts = [max(0, s - 1) for s in start_lens]" in src
    assert "want_n = len(seq) - s" in src and "want_n + 1" in src
    assert "got[1:]" in src
    # and the docstring warns rather than claiming the positions coincide.
    assert "DIFFERENT POSITIONS" in src


def test_probe_tk_creates_the_slot_k_row():
    src = body("probe_tk.py", "main")
    seq = re.search(
        r"sq\.append\(list\(prefix\) \+ list\(p\[: k - m\]\)\s*"
        r"\+ list\(gt\[k - m: k\]\) \+ \[gt\[k\]\]\)", src)
    assert seq, "probe_tk must append a token at position len(prefix)+k"
    assert "starts.append(len(prefix) + (k - m))" in src, \
        "the weight still comes from the m revealed tokens"


def test_probe_rm_creates_the_slot_k_row():
    src = body("probe_rm.py", "ce_mixed_all")
    assert "+ list(gt[k - m: k]) + [gt[k]]" in src
    assert "starts.append(len(s) - 1 - m)" in src

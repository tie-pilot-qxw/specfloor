"""Reading recorded runs, whether a probe just wrote them or they sit gzipped
in the archive.

A live probe writes plain jsonl; measurements/ stores the same records gzipped.
Every report reads through here, so a report pointed at the archive needs no
decompression step, and a malformed line -- what two probe processes sharing an
output file leave behind -- is counted rather than skipped silently.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import pathlib

ARCHIVE = pathlib.Path(__file__).resolve().parent.parent / "measurements"


def open_text(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_jsonl(path, bad=None):
    """Records in file order. Unparseable lines increment bad[path]."""
    out = []
    with open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                if bad is not None:
                    bad[str(path)] = bad.get(str(path), 0) + 1
    return out


def domain_of(path):
    """gsm8k.rpre.jsonl.gz -> gsm8k: every probe names its output this way."""
    return os.path.basename(str(path)).split(".")[0]


def load_by_domain(pattern):
    """{domain: records} over a glob, in sorted file order, plus a malformed-
    line count per domain. Sorted order fixes the prompt encounter order, which
    is what the cluster bootstraps resample in."""
    by, bad = {}, {}
    for f in sorted(glob.glob(str(pattern))):
        errs = {}
        by.setdefault(domain_of(f), []).extend(read_jsonl(f, errs))
        if errs:
            bad[domain_of(f)] = bad.get(domain_of(f), 0) + sum(errs.values())
    return by, bad


def archived(pattern):
    """A glob inside measurements/, for reports that default to the archive."""
    return str(ARCHIVE / pattern)


def weight(r):
    """Hajek weight: inverse inclusion probability under the stratified sampler."""
    return 1.0 / max(r.get("pi", 1.0), 1e-9)

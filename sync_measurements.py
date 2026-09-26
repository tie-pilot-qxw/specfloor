"""Mirror measurement_runs/ into measurements/ so the archive can be regenerated.

The archive is the paper's artifact: every recorded run, as the probes wrote it.
`.jsonl` and `.log` are stored gzipped because they are 230 MB raw and 31 MB
compressed; `.sh`, `.py` and `.md` are stored plain so they read on the web.
MANIFEST.tsv carries the raw SHA-256 of every file, so a decompressed copy can
be checked against what was measured.

    python sync_measurements.py /path/to/measurement_runs        # everything
    python sync_measurements.py /path/to/measurement_runs rpre_o1_m1024

Idempotent: a file whose raw digest already matches the manifest is left alone,
so re-running after one new run rewrites one directory rather than all of them.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEST = HERE / "measurements"
GZIP = (".jsonl", ".log")
PLAIN = (".sh", ".py", ".md", ".tsv", ".txt")
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}

# Runs the archive does not carry, because a later run covers the same anchors
# under the same configuration and the archive should hold one answer per
# question, not a history of them.
SKIP_PATHS = ("t1/", "scale/gemma12b/", "scale/qwen14b/arena8k.",
              "scale/qwen14b/C0/arena8k.")

# probe_tk's m>=1 columns are carried only by t1_fix/ and g16/. Elsewhere the
# probe was run with --rungs 0,1 for the order-0 column alone, and the order-1
# fields those runs also emitted are dropped rather than shipped unused.
DROP_ORDER1 = ("tk20/", "scale/")


def digest(path: pathlib.Path) -> tuple[str, int, int]:
    """(sha256, bytes, newline-terminated rows) of the file as it sits on disk."""
    h, n, rows = hashlib.sha256(), 0, 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
            rows += chunk.count(b"\n")
    return h.hexdigest(), n, rows


def load_manifest() -> dict[str, list[str]]:
    out = {}
    mf = DEST / "MANIFEST.tsv"
    if mf.exists():
        for line in mf.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) == 5:
                out[f[0]] = f
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = pathlib.Path(sys.argv[1]).resolve()
    only = sys.argv[2:]
    man = load_manifest()
    seen, wrote, kept = set(), 0, 0

    for path in sorted(src.rglob("*")):
        if not path.is_file() or any(p in SKIP_DIRS for p in path.parts):
            continue
        rel = path.relative_to(src)
        if any(str(rel).startswith(p_) for p_ in SKIP_PATHS):
            continue
        if only and not any(str(rel).startswith(o) for o in only):
            continue
        suffix = "".join(pathlib.Path(rel.name).suffixes)
        if not (any(s in suffix for s in GZIP) or rel.suffix in PLAIN):
            continue
        # A superseded or corrupt file keeps its trailing tag; store it as-is.
        gz = any(s in suffix for s in GZIP)
        out = DEST / (str(rel) + ".gz" if gz else str(rel))
        key = str(rel)
        seen.add(key)
        sha, raw, rows = digest(path)
        if man.get(key, [None] * 5)[4] == sha and out.exists():
            kept += 1
            man[key][3] = str(out.stat().st_size)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        drop = (str(rel).startswith(DROP_ORDER1) and rel.name.endswith(".jsonl"))
        if drop:
            buf = []
            with open(path) as fi:
                for line in fi:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    for fld in ("T", "T_split", "resid", "ess"):
                        if isinstance(rec.get(fld), dict):
                            rec[fld].pop("1", None)
                    buf.append(json.dumps(rec) + "\n")
            with gzip.GzipFile(out, "wb", 9, mtime=0) as fo:
                fo.write("".join(buf).encode())
        elif gz:
            # mtime=0 so an unchanged input produces a byte-identical archive.
            with open(path, "rb") as fi, gzip.GzipFile(out, "wb", 9, mtime=0) as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copyfile(path, out)
        # The digest recorded is of what the archive HOLDS, so a decompressed
        # copy can always be checked against MANIFEST.tsv.
        if drop:
            h = hashlib.sha256()
            with gzip.open(out, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    h.update(chunk)
            sha, raw = h.hexdigest(), sum(len(b.encode()) for b in buf)
            rows = len(buf)
        man[key] = [key, str(rows) if path.suffix == ".jsonl" else "",
                    str(raw), str(out.stat().st_size), sha]
        wrote += 1

    if not only:                                   # full sync: drop stale rows
        man = {k: v for k, v in man.items() if k in seen}
    rows = "\n".join("\t".join(man[k]) for k in sorted(man))
    (DEST / "MANIFEST.tsv").write_text(
        "path\trows\tbytes_raw\tbytes_stored\tsha256_raw\n" + rows + "\n")
    stored = sum(int(v[3]) for v in man.values())
    print(f"{len(man)} files in the archive, {wrote} written, {kept} unchanged, "
          f"{stored / 1e6:.1f} MB stored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

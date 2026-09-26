"""Make a DeepSpec DSpark checkpoint readable by the SGLang runtime in serving/.

DeepSpec writes the short-convolution and slot-embedding settings as flat top-level
keys; SGLang reads them from a `dflash_config` dict and treats a missing entry as
"module off".  A checkpoint served without this step would load, run, and report a
plausible accepted length for a model with those modules silently absent.  This
script adds the `dflash_config` entries and refuses checkpoints that ask for
something SGLang cannot serve.

    python scripts/serve/bridge_ckpt_config.py <checkpoint>           # report
    python scripts/serve/bridge_ckpt_config.py <checkpoint> --write   # apply

--write keeps the original at config.json.pre_bridge.  For the DFlash2 reproduction
(speculators format) use convert_speculators_config.py instead.
"""
import argparse
import json
import os
import shutil
import sys

SERVED_HEADS = ("vanilla", "gated", "rnn", "attn")
ATTN_HEAD_KEYS = ("markov_num_heads", "markov_head_dim", "markov_mlp_hidden",
                  "markov_gate_mode")


def plan(cfg):
    """Return (dflash_config additions, blocking problems, notes)."""
    add, blocking, notes = {}, [], []

    if "model_type" not in cfg:
        if "speculators_model_type" in cfg or "speculators_config" in cfg:
            blocking.append("speculators-format config: use convert_speculators_config.py")
        else:
            blocking.append("no model_type: transformers cannot identify this config")

    head = str(cfg.get("markov_head_type", "vanilla")).lower()
    if int(cfg.get("markov_rank", 0)) > 0:
        if head not in SERVED_HEADS:
            blocking.append(f"markov_head_type={head!r} has no SGLang head")
        else:
            notes.append(f"markov head {head!r} (rank {cfg.get('markov_rank')})")
        if head == "attn":
            # SGLang builds the head from these top-level keys; a missing one would be
            # defaulted to a different geometry than the checkpoint's.
            for key in ATTN_HEAD_KEYS:
                if key not in cfg:
                    blocking.append(f"attn head needs {key} in the checkpoint config")
    if cfg.get("slot_embed"):
        add["slot_embed"] = True
        notes.append("slot_embed: learned masked-slot embeddings")
    if cfg.get("short_conv"):
        add["short_conv_kernel_size"] = 2          # the two-tap operator
        add["short_conv_group_size"] = int(cfg.get("short_conv_group_size", 16))
        notes.append("short conv: four modules per layer")
    return add, blocking, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    path = os.path.join(args.checkpoint, "config.json")
    with open(path) as fh:
        cfg = json.load(fh)
    add, blocking, notes = plan(cfg)

    print(f"{args.checkpoint}")
    for note in notes:
        print(f"  {note}")
    existing = dict(cfg.get("dflash_config") or {})
    merged = {**existing, **add}
    missing = {k: v for k, v in add.items() if existing.get(k) != v}
    if missing:
        print(f"  dflash_config needs: {missing}")
    elif add:
        print(f"  dflash_config already has: {add}")
    for problem in blocking:
        print(f"  UNSERVABLE: {problem}")
    if blocking:
        return 1
    if merged == existing:
        print("  -> servable; config already complete")
        return 0
    if not args.write:
        print("  -> run again with --write to apply")
        return 0
    backup = path + ".pre_bridge"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    cfg["dflash_config"] = merged
    with open(path, "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"  -> wrote dflash_config={merged} (backup at {os.path.basename(backup)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

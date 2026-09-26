"""Convert the DFlash2 reproduction's speculators-format config for SGLang.

The DFlash2 reproduction is trained with speculators' own DFlash2DraftModel, so its
checkpoint is saved in the speculators schema (`speculators_model_type`, a nested
`transformer_layer_config`, no `model_type`), which SGLang cannot load.  This flattens
the transformer fields and moves the DFlash2 fields where SGLang reads them:

  * `conv_kernel_size`, `conv_group_size`, `selector_rank`, `selector_top_k` and
    `sample_from_anchor` go under `dflash_config` (SGLang reads them only there; a
    flat placement would silently serve a plain DFlash with those modules off);
  * the target taps change convention: speculators' aux_hidden_state_layer_ids
    [2, 10, 18, 26, 34] index the hidden-state list, SGLang's target_layer_ids
    [1, 9, 17, 25, 33] index decoder layers.  The -1 is checked against the
    released DFlash config for the same target rather than assumed.

With --write the original is kept as config.speculators.json and, if SGLang is
importable, the result is read back through SGLang's own parser to assert the
convolution and selector are on.
"""

import argparse
import json
import pathlib
import shutil
import sys

# The released deepseek DFlash for Qwen3-4B, which this sglang serves correctly today.
# Used as the reference tap list, not as a source of anything else.
REFERENCE_TAPS = {
    ("qwen3", 2560, 5): [1, 9, 17, 25, 33],
}
# Layer count of the TARGET, which the draft config has to state and the speculators
# export does not carry. Keyed the same way.
REFERENCE_NUM_TARGET_LAYERS = {
    ("qwen3", 2560, 5): 36,
}


def convert(cfg: dict) -> tuple[dict, list[str]]:
    notes = []
    if "speculators_model_type" not in cfg and "model_type" in cfg:
        raise SystemExit("this config is already in the sglang schema; nothing to do")

    inner = cfg.get("transformer_layer_config")
    if not isinstance(inner, dict):
        raise SystemExit("no transformer_layer_config: not a speculators draft export")

    out = dict(inner)
    out.pop("architectures", None)
    notes.append(f"flattened transformer_layer_config ({len(inner)} keys)")

    model_type = out.get("model_type")
    hidden = int(out.get("hidden_size", 0))
    layers = int(out.get("num_hidden_layers", 0))
    key = (model_type, hidden, layers)
    if key not in REFERENCE_TAPS:
        raise SystemExit(
            f"no reference tap list for {key}; add one from a config this sglang "
            f"already serves before converting, rather than trusting the -1."
        )

    taps = cfg.get("aux_hidden_state_layer_ids")
    if not taps:
        raise SystemExit("no aux_hidden_state_layer_ids to convert")
    shifted = [int(t) - 1 for t in taps]
    if shifted != REFERENCE_TAPS[key]:
        raise SystemExit(
            f"tap conversion failed its check: aux_hidden_state_layer_ids={taps} "
            f"minus one is {shifted}, but the reference config for {key} taps "
            f"{REFERENCE_TAPS[key]}. The off-by-one convention is not what this "
            f"assumed; serving it would read the wrong layers and still produce a "
            f"number."
        )
    out["target_layer_ids"] = shifted
    out["num_target_layers"] = REFERENCE_NUM_TARGET_LAYERS[key]
    notes.append(f"taps {taps} -> {shifted} (checked against the reference config)")

    # `architectures` is what picks the model class, and the two candidates differ in
    # what they BUILD, not in what they validate: Qwen3DSparkModel (the name the released
    # DFlash carries) is unconditionally a markov model and refuses markov_rank=0, while
    # DFlash2DraftModel is the class that constructs the candidate selector these weights
    # were trained with.  markov_rank stays declared at 0 to match the released config.
    out["markov_rank"] = 0
    out["architectures"] = ["DFlash2DraftModel"]

    # Read out of `dflash_config` only. Flat here means the module is not built.
    nested = ("conv_kernel_size", "conv_group_size", "selector_rank",
              "selector_top_k", "sample_from_anchor", "output_multiplier")
    # Read out of `dflash_config` OR the top level; write both so either reader agrees.
    either = ("block_size", "mask_token_id", "tie_word_embeddings")
    dflash_config = {"target_layer_ids": shifted,
                     "num_target_layers": REFERENCE_NUM_TARGET_LAYERS[key]}
    for k in nested + either:
        if k in cfg:
            dflash_config[k] = cfg[k]
    for k in either:
        if k in cfg:
            out[k] = cfg[k]
    out["dflash_config"] = dflash_config
    if "draft_vocab_size" in cfg:
        out["vocab_size"] = int(cfg["draft_vocab_size"])
    notes.append("dflash_config: " + ", ".join(
        f"{k}={dflash_config[k]}" for k in nested if k in dflash_config))

    dropped = sorted(set(cfg) - set(nested) - set(either) - {
        "transformer_layer_config", "aux_hidden_state_layer_ids", "draft_vocab_size",
        "architectures", "dtype", "transformers_version",
    })
    if dropped:
        notes.append("DROPPED (no sglang consumer): " + ", ".join(dropped))
    return out, notes


def verify(path):
    """Read the written file back through SGLang's own parser."""
    import json as _json

    try:
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
    except ImportError:
        return "readback skipped: sglang (serving/ runtime) is not importable here"

    cfg = _json.loads(path.read_text())
    from sglang.srt.models.dflash import EntryClass

    names = {c.__name__ for c in EntryClass}
    for arch in cfg["architectures"]:
        if arch not in names:
            raise SystemExit(
                f"readback: architectures={cfg['architectures']} names no class this "
                f"sglang's dflash entrypoint registers ({sorted(names)}); the loader "
                f"would pick a different draft model than the weights were trained for."
            )
    d = parse_dflash_draft_config(draft_hf_config=cfg)
    for field in ("conv_kernel_size", "conv_group_size", "selector_rank",
                  "selector_top_k"):
        if not getattr(d, field):
            raise SystemExit(
                f"readback: sglang parsed {field}=0 from the written config, so it "
                f"would serve this DFlash2 checkpoint with that module absent."
            )
    return (f"readback via sglang's parser: conv {d.conv_kernel_size}x"
            f"{d.conv_group_size}, selector rank {d.selector_rank} top_k "
            f"{d.selector_top_k}, block_size {d.block_size}, taps "
            f"{d.target_layer_ids}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    d = pathlib.Path(args.ckpt)
    src = d / "config.json"
    cfg = json.loads(src.read_text())
    out, notes = convert(cfg)

    print(d)
    for n in notes:
        print(f"  {n}")
    if not args.write:
        print("  (dry run; pass --write to apply)")
        return
    backup = d / "config.speculators.json"
    if not backup.exists():
        shutil.copy2(src, backup)
        print(f"  original kept at {backup.name}")
    src.write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"  wrote {src.name}: model_type={out['model_type']} "
          f"block_size={out.get('block_size')} taps={out['target_layer_ids']}")
    print("  " + verify(src))


if __name__ == "__main__":
    sys.exit(main())

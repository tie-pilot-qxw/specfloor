"""Summarise equivalence results as a Markdown table.

    python drafter/equivalence/summarize.py drafter/equivalence/results > summary.md
"""
import json
import sys
from pathlib import Path

ORDER = [
    "attnconv_qwen3_4b_b7_10ep",
    "dspark_qwen3_4b_b7_1ep",
    "official_dflash2_qwen3_4b_b8_1ep",
    "dspark_qwen3_4b_b7_1ep_shortconv",
    "slotembed_qwen3_4b_b7",
    "attnhead_qwen3_4b_b7",
    "attnconv_qwen3_4b_b7",
]


def main():
    root = Path(sys.argv[1])
    print("| config | init | strict load | micro-batches at init | micro-batches at checkpoint | result |")
    print("|---|---|---|---|---|---|")
    for name in ORDER:
        p = root / f"{name}.compare.json"
        if not p.exists():
            print(f"| `{name}` | | | | | not run |")
            continue
        r = json.loads(p.read_text())
        s = r["sections"]

        def cell(key):
            sec = s.get(key)
            if sec is None:
                return "-"
            ok = "equal" if sec["equal"] else "DIFFERENT"
            extra = ""
            if "losses" in sec:
                extra = f" (losses {', '.join(f'{x:.4f}' for x in sec['losses'][1])})"
            return f"{ok}, {sec['compared']} compared{extra}"

        load = s.get("load", {})
        clean = load.get("strict_clean", [None, None])[1]
        ref_load = json.loads((root / f"{name}.orig.json").read_text()).get("load", {})
        load_cell = ("equal" if load.get("equal") else "DIFFERENT") + (
            ", strict" if clean else
            f"; both trees: missing {ref_load.get('missing')} (restored from the target)")
        init_ok = s["init"]["equal"] and s["init_buffers"]["equal"] and s["trainable"]["equal"]
        init_cell = f"{'equal' if init_ok else 'DIFFERENT'}, {s['init']['compared']} tensors"
        print(f"| `{name}` | {init_cell} | {load_cell} | {cell('run_init')} | "
              f"{cell('run_checkpoint')} | "
              f"{'bit-identical' if r['equal'] else 'DIFFERENT'} |")
    ctl = root / "attnconv_qwen3_4b_b7_10ep.compare_indep.json"
    if ctl.exists():
        r = json.loads(ctl.read_text())
        parts = []
        for run in ("run_init", "run_checkpoint"):
            sec = r["sections"].get(run)
            if sec is None:
                continue
            grads = [d for d in sec["different"] if d.startswith("grad.")]
            other = [d for d in sec["different"] if not d.startswith("grad.") and d != "grad_norm"]
            a, b = sec["grad_norm"]
            parts.append(f"{run}: {len(grads)} parameter gradients differ, gradient norm "
                         f"{a:.7g} vs {b:.7g}"
                         + (f", also {other}" if other else ", losses/outputs/metrics equal"))
        print()
        print("Control, the overlay compiled independently (final configuration): "
              + ("bit-identical." if r["equal"] else "; ".join(parts) + "."))
    one = root / "onestep_attnconv_qwen3_4b_b7_10ep.json"
    if one.exists():
        r = json.loads(one.read_text())
        print()
        print("One optimizer step through train.py (final configuration): saved weights "
              + ("bit-identical" if r["bit_identical_weights"] else
                 f"DIFFERENT in {len(r['different_tensors'])} tensors")
              + f" ({r['orig']['tensors']} tensors); {r['scalars_compared']} logged scalars "
              + ("all equal" if not r["scalars_different"] else
                 f"differ: {r['scalars_different']}")
              + f" (loss {r['overlay']['loss']}, grad_norm {r['overlay']['grad_norm']}); "
              + f"the step changed {r.get('tensors_changed_by_the_step', '?')} of the "
              + f"{r['orig']['tensors']} saved tensors.")


if __name__ == "__main__":
    main()

"""Lazy access to DeepSpec, which is needed only to *run* a drafter.

The two halves of this package have different dependencies, and the split is
not an accident of packaging -- it is the same split the measurement itself
makes.

Measuring an information floor $T^{(m)}$ needs the **target alone**: sample
continuations, form the realisation family, solve the minimisation. No drafter
is instantiated and nothing but `transformers` (or an HTTP endpoint) is
required. Measuring a **model gap** $G = R - T^{(m)}$ additionally needs a real
drafter's proposal $q$, which means loading a DFlash/DSpark checkpoint and
running its block forward pass with the serving attention mask -- and that code
lives in DeepSpec.

So `probe_rpre`, `probe_br` and `eval_nll` reach for DeepSpec and nothing else
does. Rather than making the whole package unimportable when it is absent, the
symbols are resolved on first attribute access and a missing install fails with
an instruction instead of a traceback from six frames down.

Set ``SPECFLOOR_DEEPSPEC`` to a checkout path if DeepSpec is not on ``sys.path``.
"""

from __future__ import annotations

import os
import sys

_HELP = """\
This probe measures a DRAFTER's risk R, which requires running a DFlash/DSpark
block forward pass, and that lives in DeepSpec:

    git clone https://github.com/deepseek-ai/DeepSpec
    export SPECFLOOR_DEEPSPEC=/path/to/DeepSpec      # or pip install -e it

Measuring the information floor T^(m) needs none of this -- it uses the target
model alone. If you only want floors, use the probes that do not import this
module: probe_tk, probe_cheap, probe_kmedian, probe_api_floor, probe_rm, and
every *_report module.
"""

_NAMES = {
    "Qwen3DSparkModel": "deepspec.modeling.dspark",
    "Gemma4DSparkModel": "deepspec.modeling.dspark",
    "create_dspark_attention_mask": "deepspec.modeling.dspark.common",
    "create_position_ids": "deepspec.modeling.dspark.common",
    "extract_context_feature": "deepspec.modeling.dspark.common",
    # An ATTENTION markov head carries a prefix K/V cache instead of a lookup table,
    # so probe_rpre has to construct and reshape one; see head_context there.
    "AttnHeadContext": "deepspec.modeling.dspark.attn_head",
}


def _ensure_path() -> None:
    root = os.environ.get("SPECFLOOR_DEEPSPEC")
    if root and root not in sys.path:
        sys.path.insert(0, root)


def __getattr__(name: str):
    """PEP 562 module-level lazy attribute access."""
    module = _NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _ensure_path()
    try:
        mod = __import__(module, fromlist=[name])
    except ImportError as exc:                      # pragma: no cover - env dependent
        raise SystemExit(f"{exc}\n\n{_HELP}") from exc
    value = getattr(mod, name)
    globals()[name] = value                         # cache; later lookups skip __getattr__
    return value


def available() -> bool:
    """True if a drafter can be loaded. For tests and for reporting, never for
    silently degrading a measurement: a probe that needs a drafter and cannot
    find one must fail, not fall back to something that looks like a number."""
    _ensure_path()
    try:
        __import__("deepspec.modeling.dspark", fromlist=["Qwen3DSparkModel"])
    except ImportError:
        return False
    return True


def drafter_class(model_type: str):
    """The DSpark class for a target family, resolved lazily.

    Kept as a function rather than a module-level dict because the dict would
    force the import at module-import time, which is exactly what this file
    exists to avoid.
    """
    table = {"qwen3": "Qwen3DSparkModel", "gemma4_text": "Gemma4DSparkModel"}
    name = table.get(model_type)
    return None if name is None else __getattr__(name)


def drafter_families() -> list[str]:
    """The target families a drafter class exists for. No import required."""
    return ["qwen3", "gemma4_text"]

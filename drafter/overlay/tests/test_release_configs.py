"""The paper configurations load, name the right trainer, and need no optional
dependency to import (speculators is imported lazily by the DFlash2 trainer only)."""
import importlib
import sys
from pathlib import Path

import pytest

from deepspec.utils import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "dspark"
PAPER_CONFIGS = {
    "attnconv_qwen3_4b_b7_10ep": "Qwen3DSparkOnlineTrainer",
    "attnconv_qwen3_4b_b7": "Qwen3DSparkOnlineTrainer",
    "attnhead_qwen3_4b_b7": "Qwen3DSparkOnlineTrainer",
    "dspark_qwen3_4b_b7_1ep": "Qwen3DSparkOnlineTrainer",
    "dspark_qwen3_4b_b7_1ep_shortconv": "Qwen3DSparkOnlineTrainer",
    "slotembed_qwen3_4b_b7": "Qwen3DSparkOnlineTrainer",
    "official_dflash2_qwen3_4b_b8_1ep": "OfficialDFlash2Trainer",
}


@pytest.mark.parametrize("name,trainer", sorted(PAPER_CONFIGS.items()))
def test_config_loads_and_names_its_trainer(name, trainer):
    cfg = load_config(str(CONFIG_DIR / f"{name}.py"))
    assert cfg.train.trainer_cls.__name__ == trainer
    assert int(cfg.train.global_batch_size) == 512
    assert float(cfg.train.lr) == 6.0e-4
    assert cfg.model.target_model_name_or_path == "Qwen/Qwen3-4B"


def test_importing_the_package_does_not_import_speculators():
    for mod in [m for m in sys.modules if m == "speculators" or m.startswith("speculators.")]:
        del sys.modules[mod]
    importlib.import_module("deepspec.trainer")
    importlib.import_module("deepspec.trainer.official_dflash2_trainer")
    load_config(str(CONFIG_DIR / "official_dflash2_qwen3_4b_b8_1ep.py"))
    assert "speculators" not in sys.modules


def test_missing_speculators_fails_with_the_pinned_install(monkeypatch):
    import builtins

    from deepspec.trainer import official_dflash2_trainer as o

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "speculators" or name.startswith("speculators."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=o.SPECULATORS_COMMIT):
        o._import_speculators()

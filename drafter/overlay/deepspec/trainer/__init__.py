from .base_trainer import BaseTrainer
from .dspark_trainer import Gemma4DSparkTrainer, Qwen3DSparkTrainer
from .dspark_online_trainer import OnlineTargetTrainer, Qwen3DSparkOnlineTrainer
from .eagle3_trainer import Gemma4Eagle3Trainer, Qwen3Eagle3Trainer

# OfficialDFlash2Trainer (deepspec.trainer.official_dflash2_trainer) is not imported
# here: it is only needed for the DFlash2 reproduction and requires `speculators`.

__all__ = [
    "BaseTrainer",
    "Gemma4Eagle3Trainer",
    "Gemma4DSparkTrainer",
    "OnlineTargetTrainer",
    "Qwen3Eagle3Trainer",
    "Qwen3DSparkOnlineTrainer",
    "Qwen3DSparkTrainer",
]

"""One-epoch "DFlash2 reproduction" (tab:solution-components).

speculators' own DFlash2DraftModel and objective (Speculators recipe) trained on
our regenerated corpus: block size 8 with the anchor as context (7 draft slots),
top-16 candidate selector of rank 256, conv kernel 2 / group 16, CE 0.1 + TV 0.9.

Requires the optional `speculators` package (see drafter/README.md).
"""
import os

from deepspec.trainer.official_dflash2_trainer import OfficialDFlash2Trainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
# Output of scripts/data/tokenize_official_rollouts.py (see drafter/README.md).
TRAIN_DATA = os.environ.get(
    "DSPARK_TRAIN_DATA",
    "refine_data/qwen3_4b_official_pb95_multiturn_temp07_max4096/online_train.jsonl",
)

project_name = "deepspec"
exp_name = "official_dflash2_b8_qwen3_4b_1ep"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=8,
    sample_from_anchor=False,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=512,
    loss_decay_gamma=4.0,
    loss_fn='{"ce": 0.1, "tv": 0.9}',
    per_position_loss_weight="fixed-exp-decay",
    conv_kernel_size=2,
    conv_group_size=16,
    selector_rank=256,
    selector_top_k=16,
    selector_loss_alpha=1.0,
    context_window=None,
    sliding_window_non_causal=True,
)

train = dict(
    trainer_cls=OfficialDFlash2Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=2616,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
    tv_metric_stride=20,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=250,
)

data = dict(
    target_cache_path=None,
    train_data_paths=[TRAIN_DATA],
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)
def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg

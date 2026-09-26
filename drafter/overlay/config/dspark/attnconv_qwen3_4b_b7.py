"""One-epoch "Head + our convolution + slot embeddings" (tab:solution-components).

The combined variant at the one-epoch budget (2,616 steps).  Unlike the final
drafter it keeps the head's per-block anchor key/value column (markov_anchor_kv
defaults to True) and uses per-micro-batch normalisation.
"""
import os

from deepspec.trainer import Qwen3DSparkOnlineTrainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
# Output of scripts/data/tokenize_official_rollouts.py (see drafter/README.md).
TRAIN_DATA = os.environ.get(
    "DSPARK_TRAIN_DATA",
    "refine_data/qwen3_4b_official_pb95_multiturn_temp07_max4096/online_train.jsonl",
)

project_name = "deepspec"
exp_name = "attnconv_b7_qwen3_4b"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=512,
    markov_rank=512,
    markov_head_type='attn',
    markov_num_heads=4,
    markov_head_dim=128,
    markov_mlp_hidden=2048,
    markov_gate_mode='none',
    markov_out_scale=0.35,
    markov_seed_std_pred=0.0602,
    markov_seed_std_succ=0.0664,
    short_conv=True,
    short_conv_group_size=16,
    slot_embed=True,
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
    nominate_alpha=0.1,
    nominate_topk=16,
    shared_init_ablate=["markov_head", "short_conv", "slot_embed"],
    shared_init_markov_rank=256,
)

train = dict(
    trainer_cls=Qwen3DSparkOnlineTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=2616,
    lr_schedule_steps=2616,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
    tv_metric_stride=20,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=500,
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

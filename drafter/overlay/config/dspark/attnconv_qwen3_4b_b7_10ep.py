"""Final drafter ("Ours" in tab:solution-serving): 10 epochs, 26,160 steps.

Five-layer Qwen3-4B DSpark backbone with the prefix-attention head
(markov_head_type='attn': rank 512, 4 x 128 heads, SwiGLU 2048, no gate, output
scale 0.35, no anchor column), the short convolution, slot embeddings, and the
candidate-nomination loss (nominate_alpha 0.1, top-16).  The throughput switches
(fused_target, compile_l1, fsdp_ignore_frozen, window_normalized_denominator) do
not change the model; window normalisation divides by the denominator once per
optimizer step instead of once per micro-batch.

Trained on 4 GPUs (global batch 512 = 4 x 1 x 128 accumulation).  To stop early,
relaunch from a checkpoint with
    --opts train.lr_cooldown_on_resume=True --opts train.max_train_steps=<N>
which anneals the remaining steps to zero; keep it False for crash resumes.
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
exp_name = "attnconv_b7_qwen3_4b_10ep"
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
    markov_anchor_kv=False,
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
    fused_target=True,
    compile_l1=True,
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
    max_train_steps=26160,
    lr_schedule_steps=26160,
    lr_cooldown_on_resume=False,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
    tv_metric_stride=20,
    window_normalized_denominator=True,
    fsdp_ignore_frozen=True,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=654,
    keep_last_n_checkpoints=4,
    keep_optimizer_every=2616,
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

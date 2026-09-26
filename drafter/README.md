# Prefix-attention drafter: training code

Training code for the Qwen3-4B block drafter of Sec. 6 of the paper (*From diagnosis
to a prefix-attention head*) and for the one-epoch component comparison in
App. *Scope of the one-epoch component comparisons* (`tab:solution-components`).

It is shipped as an **overlay on upstream DeepSpec**
([deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) at
`afdfa7c9382a3341a3e6f17756dd816da79f132c`): `overlay/` mirrors the DeepSpec tree and
only adds or replaces files.  Serving (SGLang) lives in `../serving/`.

```
drafter/
  install.sh                 clone DeepSpec at the pinned commit and apply the overlay
  overlay/
    deepspec/modeling/dspark/attn_head.py        the prefix-attention head
    deepspec/modeling/dspark/nomination.py       candidate-nomination loss
    deepspec/modeling/dspark/slot_init.py        slot-embedding initialisation
    deepspec/modeling/dspark/qwen3/modeling.py   drafter + short conv + slot embeddings
    deepspec/modeling/dspark/qwen3/short_conv_kernel.py   fused Triton short conv
    deepspec/modeling/dspark/{common,loss,markov_head}.py, qwen3/config.py
    deepspec/modeling/fused_target.py            optional fused teacher forward
    deepspec/trainer/dspark_online_trainer.py    online-feature trainer (all DSpark rows)
    deepspec/trainer/official_dflash2_trainer.py DFlash2 reproduction (optional)
    deepspec/trainer/{base_trainer,ckpt_manager,dspark_trainer,__init__}.py
    deepspec/utils/{optim,distributed}.py
    config/dspark/*.py        the seven paper configurations
    scripts/data/             training-data regeneration and tokenisation
    scripts/train/            launch and resume
    scripts/serve/            make a checkpoint loadable by the SGLang runtime
    tests/                    unit tests (78)
  equivalence/               release-vs-research-code equivalence harness + results
```

## Install

```bash
bash drafter/install.sh ./DeepSpec          # clones upstream, checks out afdfa7c, applies overlay
cd DeepSpec
pip install -r requirements.txt              # torch, transformers, ...
python -m pytest tests -q                    # 78 tests; the fused-kernel tests need a GPU
```

The runs used Python 3.12, torch 2.9.1, transformers 5.10.2 and triton 3.5.1.
`model.fused_target=True` (used by the final run and the slot-embedding arm) needs
`sgl_kernel` (0.3.21 was used); it only changes throughput.

**Optional**: the *DFlash2 reproduction* row trains speculators' own model and
loss, so it needs `speculators` at the commit it was run with.  Nothing else
imports it:

```bash
pip install --no-deps "speculators @ git+https://github.com/vllm-project/speculators@0a1b3e0a15d67d551041933529c2c41032f5b28d"
pip install hs-connectors loguru pydantic-settings
```

## The model

Five-layer DSpark backbone on Qwen3-4B (hidden width 2560, target features from
layers `[1, 9, 17, 25, 33]`, block size 7), plus:

| component | where | configuration |
|---|---|---|
| prefix-attention head | `attn_head.py`, `markov_head_type="attn"` | residual width 512, 4 heads x 128, SwiGLU 2048, output RMS scale init 0.35, no gate, no anchor column; W2 seeded from the target lm_head (std 0.0664) |
| candidate nomination | `nomination.py`, `nominate_alpha` | weight 0.1 on CE(u; target top-16 renormalised), full-vocabulary softmax |
| short convolution | `DSparkShortConv`, `short_conv=True` | two taps, block-causal, groups of 16 channels, before and after each sublayer (20 modules, 16,486,400 parameters), identity init |
| slot embeddings | `slot_embed=True` | 6 x 2560 learned offsets on masked slots 1..6 |

Objective: DSpark's CE 0.1 + full-vocabulary L1 (TV) 0.9 + confidence BCE 1.0, plus
nomination 0.1, all with slot weights exp(-k/4).  At serving time candidates are
nominated from the base logits and the head scores transitions among the top 16
(`../serving/`).

## Configurations and checkpoints

All configurations: seed 42, global batch 512 (local batch 1), lr 6e-4 with 4%
warmup and cosine decay, gradient clipping 1.0, no weight decay, bf16, 512 anchors
per sequence, max length 4096.  One epoch is 2,616 steps.

| paper row | config (`config/dspark/`) | steps | checkpoint (Hugging Face) |
|---|---|---|---|
| Ours (`tab:solution-serving`, `tab:solution-trajectory`) | `attnconv_qwen3_4b_b7_10ep.py` | 26,160 | *(withheld for review)* `step_26160/` (also at the repo root); every epoch in `step_<2616 n>/` |
| Vanilla DSpark | `dspark_qwen3_4b_b7_1ep.py` | 2,616 | *(withheld for review)* `dspark_b7_qwen3_4b_1ep/step_2616/` |
| DFlash2 reproduction | `official_dflash2_qwen3_4b_b8_1ep.py` | 2,616 | same repo, `official_dflash2_b8_qwen3_4b_1ep/step_2616/` |
| Vanilla + our short convolution | `dspark_qwen3_4b_b7_1ep_shortconv.py` | 2,616 | same repo, `dspark_b7_qwen3_4b_1ep_shortconv/step_2616/` |
| Vanilla + slot embeddings | `slotembed_qwen3_4b_b7.py` | 2,616 | same repo, `slotembed_b7_qwen3_4b/step_2616/` |
| Prefix-attention head | `attnhead_qwen3_4b_b7.py` | 2,616 | same repo, `attnhead_b7_qwen3_4b/step_2616/` |
| Head + our convolution + slot embeddings | `attnconv_qwen3_4b_b7.py` | 2,616 | same repo, `attnconv_b7_qwen3_4b/step_2616/` |

The Hub copies of these eight `model.safetensors` files were checked byte for byte
against the checkpoints the paper's numbers were measured on.  The configurations
are identical to the `train_config.py` saved in each checkpoint apart from comments,
the data path and one no-op key (`length_balanced_batches=False`).

Differences between rows that are not in the row name (all stated in the configs):
the head variant uses the output-dependent gate (`markov_gate_mode="state_output"`)
and the vanilla-matched initial output scale; both one-epoch head variants keep the
per-block anchor key/value column, which the final run removes
(`markov_anchor_kv=False`) after it caused a divergence at step 1,283 of a first
10-epoch attempt; the final run normalises the loss once per optimizer step
(`window_normalized_denominator`), which changes the objective slightly; and the
final and slot-embedding runs use the fused teacher forward and compiled L1, which
change only the reduction order.  `shared_init_ablate` draws the shared parameters
from a sibling model without the added modules, so every arm starts from the same
backbone initialisation as vanilla DSpark.

## Training data

Open PerfectBlend (`mlabonne/open-perfectblend`, revision
`af60f3c18201652a83a93f46fcfee1b646ba3df7`) with every assistant turn regenerated
by Qwen3-4B.  Counts from the run:

| stage | rows |
|---|---|
| source rows / after dropping 3 rows with no usable user turn | 1,420,909 / 1,420,906 |
| seed-42 95/5 split: train / held out | 1,349,860 / 71,046 |
| regenerated conversations (1,790,166 assistant turns) | 1,349,860 |
| tokenised, max length 4096, >= 14 supervised tokens: kept / rejected | 1,339,867 / 9,993 |

```bash
# 1. split (the flag and the revision fix the cohort)
python scripts/data/download_and_split.py --dataset-name mlabonne/open-perfectblend \
    --revision af60f3c18201652a83a93f46fcfee1b646ba3df7 --test-size 0.05 --seed 42 \
    --drop-invalid-conversations \
    --train-output-path cache/dataset/perfectblend_train.jsonl --test-output-dir eval_datasets
# 2. shard (the run used 128 shards)
python scripts/data/shard_jsonl.py --input cache/dataset/perfectblend_train.jsonl \
    --output-dir cache/dataset/perfectblend_train_shards_128 --num-shards 128 \
    --expected-total 1349860
# 3. regenerate each shard: one SGLang server per GPU, temperature 0.7, top-p 0.8,
#    top-k 20, min-p 0, thinking disabled, max 4096 tokens per turn, all turns in order
bash scripts/data/launch_rollout_shard.sh qwen3_4b <shard> 128 "0 1 2 3"
# 4. validate against the source ids and merge
BASE=refine_data/rollouts/qwen3_4b_official_pb95_multiturn_temp07_max4096
python scripts/data/validate_regenerated_conversations.py "$BASE"/shard_*_of_00128.jsonl \
    --source-inputs cache/dataset/perfectblend_train_shards_128/shard_*_of_00128.jsonl \
    --expected-total 1349860 --merge-out "$BASE"/merged.jsonl
# 5. tokenise with the Qwen chat template (assistant spans supervised)
python scripts/data/tokenize_official_rollouts.py --input "$BASE"/merged.jsonl \
    --output "$BASE"/online_train.jsonl --rejected-output "$BASE"/online_rejected.jsonl \
    --target Qwen/Qwen3-4B --chat-template qwen --max-length 4096 --min-loss-tokens 14 \
    --expected-total 1349860
```

The regenerated shards used for training are published as
a Hugging Face dataset *(withheld for review)*
(`data/shard_*_of_00128.jsonl`; all 128 checked byte for byte against the training
run's).  To rebuild the exact training file without regenerating, run steps 1-2,
download those shards into `$BASE`, and run steps 4-5; the merged file should have
SHA-256 `59c742412dfc2a0f5989dbd244c503d32e0d82a576c3a42bae14bcd5a994b6eb`.  Running
step 3 instead samples a new, statistically equivalent corpus.  See `DATA_CARD.md`.
Use of the generated data is subject to the terms of Open PerfectBlend and Qwen3-4B.

## Training

```bash
export DSPARK_TRAIN_DATA=$BASE/online_train.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train/train_drafter.sh config/dspark/attnconv_qwen3_4b_b7_10ep.py
```

The paper's runs used 4 H100 80GB GPUs (gradient accumulation 128); the final run
took about 15.2 s per optimizer step, 4.6 days for 10 epochs.  Any GPU count that
divides 512 gives the same 512 samples per step.  Checkpoints go to
`~/checkpoints/deepspec/<exp_name>/step_<N>`; an interrupted run resumes from
`step_latest` automatically (`scripts/train/resume_drafter.sh` checks the resume
point first).  To end a long run early, relaunch with
`--opts train.lr_cooldown_on_resume=True --opts train.max_train_steps=<N>`.

Before serving a checkpoint with the SGLang runtime in `../serving/`:

```bash
python scripts/serve/bridge_ckpt_config.py <checkpoint> --write          # DSpark rows
python scripts/serve/convert_speculators_config.py <checkpoint> --write  # DFlash2 row
```

The released checkpoints already carry the converted `config.json`.

## Equivalence with the code that produced the paper

The overlay is a trimmed extraction of a larger research tree.  `equivalence/`
checks that, for every paper configuration, the release computes exactly what the
original code computed.  `run_step.py` drives the real trainer
(`build_models` + `run_batch`) in either tree and records SHA-256 digests;
`compare.py` requires bit equality of:

* every parameter and buffer after from-scratch initialisation at the config seed
  (including the shared-initialisation sibling and the seeded W2);
* the set of missing/unexpected keys when the released checkpoint is loaded
  (the release loads every checkpoint with `strict=True`);
* per micro-batch (three training sequences, one per micro-batch as in training,
  64 anchors): the loss, every forward output (draft logits, confidence, target
  logits, masks, nomination terms), every logged metric, and after backward every
  parameter gradient and the gradient norm, both at initialisation and at the
  released weights.

`one_step.sh` additionally runs one optimizer step through `train.py` (FSDP,
torch.compile, BF16 optimizer, window normalisation, checkpoint save) and compares
the saved weights.

torch.compile's inductor picks reduction-kernel block sizes by timing on first
compilation, so two independent compilations can differ in the last bit of a
compiled reduction.  Both trees are therefore compared on the same compiled
kernels (the original tree's cache, copied); a separate control compiles
independently.  Results are in `equivalence/results/` and summarised below.

| config | init | strict load | micro-batches at init | micro-batches at checkpoint | result |
|---|---|---|---|---|---|
| `attnconv_qwen3_4b_b7_10ep` | equal, 120 tensors | equal, strict | equal, 487 compared (losses 3.0662, 0.5797, 3.7250) | equal, 487 compared (losses 0.7213, 0.0077, 0.0346) | bit-identical |
| `dspark_qwen3_4b_b7_1ep` | equal, 64 tensors | equal, strict | equal, 395 compared (losses 3.5319, 4.2872, 3.8823) | equal, 395 compared (losses 1.8398, 0.7001, 0.1943) | bit-identical |
| `official_dflash2_qwen3_4b_b8_1ep` | equal, 85 tensors | equal; both trees: missing ['verifier_lm_head.weight', 'verifier_norm.weight'] (restored from the target) | equal, 330 compared (losses 5.2186, 4.6938, 5.0141) | equal, 330 compared (losses 0.9348, 0.0781, 0.0679) | bit-identical |
| `dspark_qwen3_4b_b7_1ep_shortconv` | equal, 104 tensors | equal, strict | equal, 435 compared (losses 3.5319, 4.2872, 3.8823) | equal, 435 compared (losses 1.8757, 0.6751, 0.1630) | bit-identical |
| `slotembed_qwen3_4b_b7` | equal, 65 tensors | equal, strict | equal, 396 compared (losses 3.5408, 4.2679, 3.8295) | equal, 396 compared (losses 1.8911, 0.4915, 0.1980) | bit-identical |
| `attnhead_qwen3_4b_b7` | equal, 83 tensors | equal, strict | equal, 450 compared (losses 5.3772, 4.9643, 4.9854) | equal, 450 compared (losses 2.2046, 0.5694, 0.1800) | bit-identical |
| `attnconv_qwen3_4b_b7` | equal, 122 tensors | equal, strict | equal, 489 compared (losses 5.8479, 5.8166, 7.0822) | equal, 489 compared (losses 2.1428, 0.5614, 0.1607) | bit-identical |

Control, the overlay compiled independently (final configuration): run_init: 84 parameter gradients differ, gradient norm 75.10371 vs 75.10391, losses/outputs/metrics equal; run_checkpoint: 84 parameter gradients differ, gradient norm 0.6897429 vs 0.6897045, losses/outputs/metrics equal.

One optimizer step through train.py (final configuration): saved weights bit-identical (120 tensors); 144 logged scalars all equal (loss [4.501242160797119], grad_norm [186.0]); the step changed 89 of the 120 saved tensors.

All runs used Python 3.12, torch 2.9.1 and one H100, with three training sequences
(one per micro-batch), 64 anchors per sequence and max length 1024.  The DFlash2
row ran with speculators at the pinned commit (the same source as the documented
`pip` install).  Its checkpoint does not contain `verifier_lm_head` and
`verifier_norm`, which are copied from the target, in both trees.  The control
shows what an independent compilation changes: nothing but the last bits of
gradients behind compiled reductions.

## Not included

The research tree also contained self-draft/LoRA and XG drafters, an EAGLE-chain
variant, Qwen2 and Gemma-4 online trainers, other order-1 head types (conditional,
multi-lag, window-mix, XPress), within-block attention lanes, horizon scaling,
split cross/self attention, DFlash2-wiring convolutions, depth-credit, serving,
KL-guard and fork-weighted objectives, profiling hooks, and many unused
configurations.  None of them is used by the paper's configurations.

The equivalence checks cover model construction, the forward/backward training
step for every configuration, and one full optimizer step through `train.py` on
one GPU.  They do not exercise multi-GPU reduction (the paper's runs used 4 GPUs),
resuming from a checkpoint, or `lr_cooldown_on_resume`; that code is carried over
from the research tree with only the LoRA-specific save option removed, and
`tests/trainer/` covers its resume-state logic.

## License

MIT, as upstream DeepSpec (`LICENSE`); third-party notices in `NOTICE`.

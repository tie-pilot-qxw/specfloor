# A second and third target: Qwen3-8B and Gemma-4-12B

The whole 4B pipeline, re-run against each target's own released DFlash and DSpark
checkpoints. Same protocol, same estimators, same code; C0, γ=7, M=256 for the
gap probes and M=1024 for the floors.

```
bash measurement_runs/scale/smoke.sh  qwen8b   Qwen/Qwen3-8B       <dflash_8b>  <dspark_8b>  5
bash measurement_runs/scale/run_target.sh qwen8b   Qwen/Qwen3-8B       <dflash_8b>  <dspark_8b>  5
bash measurement_runs/scale/run_target.sh gemma12b google/gemma-4-12B-it <dflash_12b> <dspark_12b> 5
```

`smoke.sh` runs the whole chain at toy size (2 prompts, 4 anchors) first. It is
not optional: it caught all three of the setup problems below in minutes each,
rather than after a corpus stage had burned an afternoon.

## The corpus is per target and cannot be shared

A floor lives at the positions the target itself would occupy. Anchoring one
model inside another model's text measures its uncertainty about continuing
someone else's writing — out of distribution, and not the quantity that governs
its own speculative decoding. Prompts are shared so the domain axis stays
comparable; responses are not.

## Three domains, not four

arena8k is the only domain whose contexts run long: median 2.4k tokens and up to
8.5k, against at most 1.7k anywhere in gsm8k, mbpp or alpaca. At M=256 with a
full-vocabulary TV the expanded prefix cache those anchors need does not fit one
80 GiB card for the larger targets.

* 8B kept 87/96 at both 12 and 24 GiB. The 9 dropped all had longer contexts than
  every anchor kept (min 8062 against a kept max of 7667) — a clean
  length-dependent cut.
* 12B kept 57/96 at order 0, also a clean length cut (kept max 2936, dropped min
  3219), and only 6/96 at order 1, which is NOT a length cut — anchors as short
  as ctx 90 were dropped. No verified account of the order-1 case; it is recorded
  as observed rather than explained.

So the paper reports **gsm8k, mbpp and alpaca — 288 anchors per target, nothing
skipped anywhere**. Dropping the domain uniformly is the honest cut; pooling it
complete for one target and length-truncated for another is not. The floors
(probe_tk, sglang, its own memory management) are complete on all four domains
for all three targets, so only the gap side is restricted.

Raising `--kv-budget-gib` does not fix the 8B case: chunk grows with the budget,
so the peak grows with it too. 12 and 24 GiB skip the same 9 anchors.

## Result 1: the scaling objection fails, and not monotonically

order 0 (DFlash), three domains, 288 anchors each:

| slot | T 4B | R 4B | G/R 4B | T 8B | R 8B | G/R 8B | T 12B | R 12B | G/R 12B |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0000 | 0.1144 | 100% | 0.0000 | 0.1646 | 100% | 0.0000 | 0.2077 | 100% |
| 1 | 0.0593 | 0.2292 | 74.1% | 0.1046 | 0.2706 | 61.4% | 0.0840 | 0.4100 | 79.5% |
| 2 | 0.1074 | 0.3231 | 66.8% | 0.1748 | 0.4002 | 56.3% | 0.1592 | 0.4889 | 67.4% |
| 3 | 0.1613 | 0.4156 | 61.2% | 0.2306 | 0.4468 | 48.4% | 0.2220 | 0.6241 | 64.4% |
| 4 | 0.1988 | 0.4789 | 58.5% | 0.2795 | 0.5082 | 45.0% | 0.2686 | 0.6608 | 59.3% |
| 5 | 0.2331 | 0.5399 | 56.8% | 0.3228 | 0.5815 | 44.5% | 0.3102 | 0.7341 | 57.7% |
| 6 | 0.2698 | 0.5965 | **54.8%** | 0.3602 | 0.6211 | **42.0%** | 0.3315 | 0.7811 | **57.6%** |

If the gap were an artefact of drafter budget it should fall as the target grows.
The 4B→8B step does move that way — the floor rises faster than the loss, so the
gap's share drops twelve points — but the step to another family reverses it.
**The largest target has the largest gap share of the three.** Gemma-4-12B's
floor sits between the two Qwen targets at every slot while its DFlash carries
0.781 at slot 6 against their 0.621 and 0.597, so the extra loss is gap, not
information. The floor is a property of how a model writes, not of how large it is.

Cross-estimator check: probe_tk (SNIS, M=1024, top-256, no shared code path with
the table) gives T at slot 6 of 0.3609 for 8B and 0.3221 for 12B, against 0.3602
and 0.3315 above.

## Result 2: the chain drafter's headroom is the stable quantity

order 1 (DSpark), same anchors:

| slot | T1 4B | T1 8B | T1 12B | Gp/Ro 4B | 8B | 12B | expo 4B | 8B | 12B |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0010 | 0.0008 | 0.0011 | 99.3% | 99.5% | 99.6% | 0.0818 | 0.1024 | 0.0972 |
| 2 | 0.0038 | 0.0104 | 0.0118 | 98.0% | 95.3% | 95.9% | 0.1134 | 0.1620 | 0.1755 |
| 3 | 0.0121 | 0.0168 | 0.0242 | 95.3% | 92.8% | 93.6% | 0.1482 | 0.2042 | 0.1951 |
| 4 | 0.0203 | 0.0305 | 0.0224 | 91.6% | 89.1% | 94.4% | 0.1932 | 0.2161 | 0.2349 |
| 5 | 0.0257 | 0.0314 | 0.0258 | 91.4% | 90.4% | 94.3% | 0.2139 | 0.2337 | 0.2462 |
| 6 | 0.0264 | 0.0431 | 0.0357 | **92.1%** | **87.3%** | **92.4%** | 0.2317 | 0.2567 | 0.2641 |

T^(1) never exceeds 0.043 anywhere. G_post is 87–99% of the chain's
oracle-conditioned risk on all three targets, across two architecture families
and 3× in size. Exposure grows with target size in both families.

## Setup problems the smoke test caught, and what each really was

1. **`deepseek-ai/dflash_qwen3_8b_block7` was config-only in the local cache**
   (32K, no `model.safetensors`). Downloaded.

2. **sglang asserts Gemma4 out of fa3** (`server_args.py:5478-5492`), while
   deterministic mode separately requires a backend in
   `RADIX_SUPPORTED_DETERMINISTIC_ATTENTION_BACKEND` or it silently drops the
   radix cache. `triton` is the only member of both sets. `backend.py` now
   resolves the backend from the target's architectures instead of pinning fa3.
   Qwen3 still resolves to fa3, so every Qwen number is untouched.

3. **Gemma4's released config is a multimodal wrapper.** `hidden_size` and
   `vocab_size` live in `.text_config`. The first crashed; the second is the one
   worth naming, because a wrapper that answered with a different `vocab_size`
   would have produced numbers rather than an error. Both now unwrap.

## Instrumentation checks run on the Gemma result, all clean

Gemma's R is much higher than either Qwen's, so before reporting it as a property
of that drafter the three ways it could have been an artefact were checked:

* **Are the taps reading the text tower?** `AutoModelForCausalLM` resolves
  `gemma4_unified` to `Gemma4UnifiedForConditionalGeneration`, not
  `...ForCausalLM`. Its forward passes `hidden_states` straight through from
  `Gemma4UnifiedModel` → `self.language_model`, so they are the text tower's.
  Clean.
* **Is `final_logit_softcapping` applied on both sides?** The target's top-level
  config has none, but its forward reads
  `self.config.get_text_config().final_logit_softcapping` = 30.0 and applies it.
  The drafter's `compute_logits`
  (`deepspec/modeling/dspark/gemma4/modeling.py:339`) reads the same field from
  its own config and applies the same cap. Matched — a mismatch here would have
  inflated R for gemma alone.
* **Is the tap extractor the training-time one?** `extract_context_feature` is
  DeepSpec's own (`deepspec/modeling/dspark/common.py:107`), the function the
  drafter was trained against. Correct by construction.

## Not a memory problem worth more budget

The `!! OOM at ctx=` skips above are real limits at M=256, not contention: the
gemma run finished at 22:18 and the card's other tenants arrived at 22:27. A
later diagnostic *was* contended and its OOM should be ignored. Anything that
reruns arena8k for these targets needs either a smaller M on that domain or a
second card, not a larger `--kv-budget-gib`.

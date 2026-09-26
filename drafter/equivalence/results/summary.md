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

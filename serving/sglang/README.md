# SGLang runtime for the prefix-attention drafter

`sglang-dspark.patch` adds serving support for our drafter (and the one-epoch
component variants) to SGLang. It is a single diff against upstream
[sgl-project/sglang@702de2631057d11542f1930b47e84a05cdd57587](https://github.com/sgl-project/sglang/tree/702de2631057d11542f1930b47e84a05cdd57587)
(`0.5.19.dev451`), restricted to `python/` and `test/`: 27 files, +3978/−59.

| Component | Files |
|---|---|
| Prefix-attention head (`markov_head_type="attn"`): predecessor query, cross-attention over the committed prefix's projected keys/values, SwiGLU MLP; precomputed projections and a fused context-KV write | `srt/models/dspark.py`, `kernels/ops/speculative/dspark/{attn_head,fused_ctx_kv_write}.py` |
| Two-tap block-causal short convolution, four independent modules per layer; gate projection folded into the compiled region (`SGLANG_DFLASH_FUSE_CONV`); a Triton small-batch decode kernel | `srt/models/dflash.py`, `kernels/ops/speculative/dflash_short_conv.py` |
| Learned slot embeddings, block-self attention, split head counts | `srt/models/dflash.py` |
| Top-16 candidate lattice: the head scores every (predecessor, successor) pair in parallel and a walk picks the chain, folded into the CUDA graph for greedy and sampling batches; the verifier receives the lattice's q as probabilities | `srt/speculative/dspark_components/*`, `srt/speculative/dflash_*` |
| Unit and parity tests | `test/srt/speculative/test_*.py`, `test/registered/spec/dspark/test_dspark_ctx_kv_fused_parity.py` |

## Revisions

The patch is the research branch `deepspec-eval` at `81f7f3ff13db` with three
inert pieces removed (below). Each archived measurement ran on one of these:

| Measurement | Runtime | How to get it |
|---|---|---|
| Serving sweep, 2026-09-22 (Table *serving sweep*; τ and S in the main and per-task serving tables) and the draft-forward probe | `81f7f3ff13db` | `sglang-dspark.patch` |
| Sequential single-request runs, 2026-09-15 (historical, superseded; `results/historical_20260915`) | `8f51a8adf6db` | then `git apply sep15-revert-fused-ctx-kv-write.patch` |
| Training trajectory and one-epoch components, 2026-09-01 to 09-09 (`results/accept_evals`) | an earlier state of the same branch; the exact revision was not recorded | `sglang-dspark.patch` is the closest; see below |

`81f7f3f` differs from `8f51a8a` only by the fused context-KV write for the
attention head (`fused_ctx_kv_write.py`, its parity test, and the call site in
`dspark.py`); `sep15-revert-fused-ctx-kv-write.patch` undoes exactly that.

The trajectory and component evaluations predate the lattice folding into the
CUDA graph (`09e8529792`, 2026-09-09) and the September 14–15 changes to how
the lattice's q reaches the verifier. Those changes do not alter temperature-0
or temperature-1 acceptance by construction (greedy verification never reads q,
and a temperature of 1 is unaffected by the double temperature that was fixed),
but the exact runtime of those runs cannot be rebuilt. Re-running one component
(`lat_attnconv`, 430 rows, temperature 0) on this patch is reported in
`../README.md`.

### Removed from the research branch

All three were inactive in every archived run; each is gated by a switch the
runs did not set, or by a checkpoint type the paper does not use.

- `DFLASH_PROBE` ablation hook (`39f61b66c9`, 387 lines in `dflash.py`): active
  only when `DFLASH_PROBE` is set. No archived server log contains its
  activation warning.
- `cond` markov head (`1addc55994`): no paper checkpoint has
  `markov_head_type="cond"`; its test and its case in `test_markov_lattice.py`
  go with it.
- `SGLANG_DSPARK_ACCEPT_HIST` position-wise acceptance dump (`53a63832f2`,
  reverted): a no-op unless the variable names a file.

Kept although inert: the grouped short-conv kernel (`3c3632c041`) selected by
`SGLANG_DFLASH_CONV_KERNEL=1`. The same commit introduced
`SGLANG_DFLASH_FUSE_CONV`, which every sweep point used, so it is not separable.
The default `SGLANG_DFLASH_CONV_KERNEL=2` selects the decode kernel instead.

## Install

The archived runtime: Python 3.12.3, CUDA 13.0, torch 2.13.0, triton 3.7.1,
flashinfer-python 0.6.17, sglang-kernel 0.4.6.post1, transformers 5.12.1, on
H100 80GB HBM3 (driver 580.159.03). `pip-freeze.txt` lists the full
environment (SGLang itself excluded).

```bash
git clone https://github.com/sgl-project/sglang.git && cd sglang
git checkout 702de2631057d11542f1930b47e84a05cdd57587
git apply /path/to/serving/sglang/sglang-dspark.patch
pip install -e "python"          # or install 702de2631 and put this python/ first on PYTHONPATH
```

The archived runtime was the upstream wheel of `702de2631` with the changed
files copied over it, which is equivalent: every file this patch touches was
checked byte for byte against that runtime (see Verification).

## Environment switches

| Variable | Default here | Archived runs | Effect |
|---|---|---|---|
| `SGLANG_DFLASH_FUSE_CONV` | `0` | **`1`** in the sweep and the probe | Folds the short-conv gate projection into the compiled region with a Triton GEMM template. Set it to reproduce the sweep. The September 1–9 evaluations predate it and ran the unfused path. |
| `SGLANG_DSPARK_FOLDED_LATTICE` | `1` | `1` | Lattice proposal inside the CUDA graph. |
| `SGLANG_DSPARK_ATTN_PRECOMPUTE` | `1` | `1` (unset) | Precomputed attention-head projections. |
| `SGLANG_DFLASH_CONV_KERNEL` | `2` | `2` (unset) | Short-conv decode kernel; `1` is the inert grouped kernel, `0` the compiled path. |
| `SGLANG_RAGGED_VERIFY_MODE` | `static` (upstream) | `static` | |
| `SGLANG_RECORD_STEP_TIME` | `0` (upstream) | `1` in the sweep | Scheduler step timing used by the sweep's decode diagnostic. |

`../sweep/common.py` sets the four the sweep needs.

## Tests

```bash
python -m pytest -q test/srt/speculative/test_{attn_head_input_projection,attn_head_parity,attn_head_precompute,cross_self_split,dflash2_conv_parity,lattice_proposal_q,markov_lattice,short_conv_decode,short_conv_parity}.py \
    test/registered/spec/dspark/test_dspark_ctx_kv_fused_parity.py
```

Most tests need a CUDA GPU. Four parity tests compare against the training
modules and skip unless the `deepspec` package (the drafter's training code)
is importable.

## Verification

- The patch applies cleanly to a fresh `702de2631` checkout, and the Sep-15
  revert patch applies on top of it.
- Of the 17 source files it touches, 13 are byte-identical to the runtime the
  sweep ran on, apart from the one-line notice below. The other four
  (`dflash.py`, `dspark.py`, `dspark_config.py`, `dspark_verify.py`) differ only
  by the three removals above. The archived sweep manifests record the runtime
  `dspark.py` and `dspark_draft.py` hashes (`316b6110…`, `3c03773b…`), which are
  `81f7f3f`'s.
- All 55 tests above pass on an H100, with the training-side reference
  importable, so none skip.
- A temperature-0 smoke run on this patch reproduces archived sweep rows
  exactly (see `../README.md`).

## License

SGLang is licensed under Apache-2.0
([LICENSE](https://github.com/sgl-project/sglang/blob/702de2631057d11542f1930b47e84a05cdd57587/LICENSE)).
As required by its section 4(b), every file this patch adds or modifies begins
with a comment stating that it was added or modified relative to
`sgl-project/sglang@702de2631`.

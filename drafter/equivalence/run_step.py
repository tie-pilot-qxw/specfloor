"""Record bitwise digests of drafter initialisation, checkpoint loading and training
micro-batches, using whichever DeepSpec tree is given by --repo.

Run it once against the original research tree and once against upstream DeepSpec
plus drafter/overlay, then compare the two JSON files with compare.py.  It drives
the real trainer code (`build_models` and `run_batch`) but not BaseTrainer.__init__,
so there is no FSDP wrapper, optimizer or checkpoint directory.

    python drafter/equivalence/run_step.py --repo <deepspec-tree> \
        --config <tree>/config/dspark/attnconv_qwen3_4b_b7_10ep.py \
        --checkpoint <dir with model.safetensors> --data online_train.jsonl \
        --out result.json

Per configuration it records:
  init     SHA-256 of every parameter and buffer after build_models() (from-scratch
           initialisation at the config's seed, including the shared-init sibling
           and the target-seeded head codebook);
  load     missing / unexpected keys when the checkpoint is loaded with strict=True;
  steps    for each micro-batch (one sequence each, as in training): the loss bits,
           SHA-256 of every forward output tensor, all logged metrics, and after
           backward the SHA-256 of every parameter gradient and the gradient norm.
The micro-batches run twice: once at the from-scratch initialisation and once with
the checkpoint weights.
"""

import argparse
import hashlib
import json
import os
import struct
import sys


def _sha(t):
    import torch

    t = t.detach()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()


def _bits(x):
    return struct.pack(">d", float(x)).hex()


def _digest_outputs(out, prefix="out"):
    """SHA-256 of every tensor reachable from a forward output (dataclass, tuple,
    list or dict)."""
    import dataclasses

    import torch

    rec = {}
    if torch.is_tensor(out):
        rec[prefix] = _sha(out)
    elif dataclasses.is_dataclass(out):
        for f in dataclasses.fields(out):
            rec.update(_digest_outputs(getattr(out, f.name), f"{prefix}.{f.name}"))
    elif isinstance(out, dict):
        for k in sorted(out):
            rec.update(_digest_outputs(out[k], f"{prefix}.{k}"))
    elif isinstance(out, (tuple, list)):
        for i, v in enumerate(out):
            rec.update(_digest_outputs(v, f"{prefix}.{i}"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="DeepSpec tree to import")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--data", required=True, help="online_train.jsonl")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--num-anchors", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234, help="anchor-sampling seed")
    ap.add_argument("--port", type=int, default=29611)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo))
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file

    import deepspec
    from deepspec.utils import load_config, parse_opts_to_config, seed_all
    from deepspec.utils import metrics as M

    assert os.path.abspath(os.path.dirname(os.path.dirname(deepspec.__file__))) == \
        os.path.abspath(args.repo), f"imported deepspec from {deepspec.__file__}"

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl", init_method=f"tcp://127.0.0.1:{args.port}", rank=0, world_size=1,
            device_id=torch.device("cuda", 0))

    cfg = parse_opts_to_config(
        [f"model.num_anchors={args.num_anchors}", f"data.max_length={args.max_length}"],
        load_config(args.config),
    )
    cls = cfg.train.trainer_cls
    tr = cls.__new__(cls)
    tr.args = cfg
    tr.device = torch.device("cuda", 0)
    tr.precision_dtype = torch.bfloat16
    tr.world_size, tr.global_rank = 1, 0
    tr.next_micro_step = 0
    tr.gradient_accumulation_steps = max(1, args.rows)

    seed_all(int(cfg.seed))                     # as train.py does before the trainer
    draft_model, _ = tr.build_models()
    result = {
        "repo": os.path.abspath(args.repo),
        "config": os.path.abspath(args.config),
        "trainer": f"{cls.__module__}.{cls.__name__}",
        "overrides": {"num_anchors": args.num_anchors, "max_length": args.max_length},
        "init": {n: _sha(p) for n, p in draft_model.named_parameters()},
        "init_buffers": {n: _sha(b) for n, b in draft_model.named_buffers()},
        "trainable": sorted(n for n, p in draft_model.named_parameters() if p.requires_grad),
    }
    M.reset()

    rows = []
    with open(args.data) as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= args.rows:
                break
    collate = cls.data_collator_cls()

    captured = {}

    def hook(_m, _i, out):
        captured["out"] = out

    draft_model.register_forward_hook(hook)
    tr.draft_model = draft_model
    tr.model = draft_model
    draft_model.train()

    def run(tag):
        steps = []
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        for p in draft_model.parameters():
            p.grad = None
        tr.next_micro_step = 0
        for i, row in enumerate(rows):
            batch = collate([row])
            loss = tr.run_batch(batch)
            (loss / len(rows)).backward()
            out = captured.pop("out", None)
            rec = {"loss": float(loss), "loss_bits": _bits(loss)}
            rec.update(_digest_outputs(out))
            win = getattr(tr, "_window_den", None)
            if win is not None:
                rec["window_den"] = float(win)
            rec["metrics"] = {k: _bits(v) for k, v in M.flush().items()}
            steps.append(rec)
            tr.next_micro_step += 1
        grads = {n: _sha(p.grad) for n, p in draft_model.named_parameters()
                 if p.grad is not None}
        norm = torch.sqrt(sum((p.grad.double() ** 2).sum()
                              for p in draft_model.parameters() if p.grad is not None))
        return {"steps": steps, "grads": grads, "grad_norm": float(norm),
                "grad_norm_bits": _bits(norm)}

    result["run_init"] = run("init")

    if args.checkpoint:
        sd = load_file(os.path.join(args.checkpoint, "model.safetensors"))
        missing, unexpected = draft_model.load_state_dict(sd, strict=False)
        result["load"] = {"missing": sorted(missing), "unexpected": sorted(unexpected),
                          "tensors": len(sd)}
        if not missing and not unexpected:
            draft_model.load_state_dict(sd, strict=True)
        result["run_checkpoint"] = run("checkpoint")

    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print(f"wrote {args.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

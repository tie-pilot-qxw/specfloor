"""Is the DFlash forward wired correctly? Per-slot top-1 agreement with the target.

If the mask / position_ids / hidden-state taps are right, slot 0 -- where the
drafter sees the full prefix and predicts one token ahead -- should match the
target's greedy token most of the time, and the match rate should decay along
the block. A flat or low slot-0 rate means the wiring is wrong, not the model.
"""
import sys, json, random, torch
sys.path.insert(0, "/workspace/DeepSpec")
from transformers import AutoModelForCausalLM, AutoTokenizer
from measurement import config as C
from measurement.probe_rpre import load_drafter, drafter_q
from deepspec.modeling.dspark.common import extract_context_feature

DEV = "cuda"
K = C.GAMMA
seqs = {}
for line in open("/workspace/measurement_runs/calib_20260818/C0/gsm8k.jsonl"):
    r = json.loads(line); seqs[r["prompt_id"]] = r
anchors = [json.loads(l) for l in open("/workspace/measurement_runs/calib_20260818/C0/ladder.M512.jsonl")]
anchors = [a for a in anchors if a["prompt_id"] in seqs]
random.Random(C.SEED).shuffle(anchors)
anchors = anchors[:40]

target = AutoModelForCausalLM.from_pretrained(C.TARGET, dtype=torch.bfloat16,
                                              attn_implementation="sdpa").to(DEV).eval()
draft, dcfg = load_drafter("/workspace/dflash_sgl", DEV)
taps = list(dcfg.target_layer_ids)

hit_greedy = [0]*K; hit_gold = [0]*K; n = 0
acc_g = 0.0
with torch.no_grad():
    for a in anchors:
        s = seqs[a["prompt_id"]]
        full = s["prompt_ids"] + s["response_ids"]
        cut = s["prompt_len"] + a["t"]
        if len(full) < cut + K: continue
        prefix = torch.tensor([full[:cut]], device=DEV)
        gold = full[cut:cut+K]
        th = target(input_ids=prefix, output_hidden_states=True, use_cache=False)
        thid = extract_context_feature(th.hidden_states, taps).to(torch.bfloat16)
        q = drafter_q(draft, dcfg, thid, prefix, K, DEV)          # [K,V]
        dpred = q.argmax(-1).tolist()
        # target GREEDY continuation, teacher-forced on its own greedy tokens
        gseq = list(full[:cut]); greedy = []
        for _ in range(K):
            lg = target(input_ids=torch.tensor([gseq], device=DEV), use_cache=False).logits[0,-1]
            t_ = int(lg.argmax()); greedy.append(t_); gseq.append(t_)
        run = True
        for k in range(K):
            if dpred[k] == greedy[k]: hit_greedy[k] += 1
            if dpred[k] == gold[k]:   hit_gold[k]   += 1
            if run and dpred[k] == greedy[k]: acc_g += 1
            else: run = False
        n += 1

print(f"\nn={n} anchors, block K={K}")
print(f"{'slot':>4} {'match target-greedy':>20} {'match gold':>12}")
for k in range(K):
    print(f"{k:>4} {hit_greedy[k]/n:>20.3f} {hit_gold[k]/n:>12.3f}")
print(f"\nmean accepted length vs greedy (exact-match prefix) = {acc_g/n:.2f} / {K}")
print("Published DFlash block-7 is ~2.5-3.5 accepted. A slot-0 rate near 0.8+ and")
print("a decaying curve means the forward is right; a flat low curve means it is not.")

import sys, json, random, torch
sys.path.insert(0, "/workspace/DeepSpec")
from transformers import AutoModelForCausalLM
from measurement import config as C
from measurement.probe_rpre import load_drafter, drafter_q, rollout, barycentre, mean_tv
from measurement.corpus import resolve_stops
from measurement.probe_cheap import anchor_seed
from deepspec.modeling.dspark.common import extract_context_feature
from transformers import AutoTokenizer

DEV="cuda"; K=C.GAMMA
seqs={}
for l in open("/workspace/measurement_runs/calib_20260818/C0/arena8k.jsonl"):
    r=json.loads(l); seqs[r["prompt_id"]]=r
anchors=[json.loads(l) for l in open("/workspace/measurement_runs/calib_20260818/C0/arena8k.ladder.M512.jsonl")]
anchors=[a for a in anchors if a["prompt_id"] in seqs]
random.Random(C.SEED).shuffle(anchors); anchors=anchors[:96]
ctx=sorted((seqs[a["prompt_id"]]["prompt_len"]+a["t"]) for a in anchors)
print("context lens: min",ctx[0],"p50",ctx[len(ctx)//2],"p90",ctx[int(.9*len(ctx))],"max",ctx[-1])

tok=AutoTokenizer.from_pretrained(C.TARGET)
stop_ids=resolve_stops(C.TARGET,tok)
target=AutoModelForCausalLM.from_pretrained(C.TARGET,dtype=torch.bfloat16,attn_implementation="sdpa").to(DEV).eval()
draft,dcfg=load_drafter("/workspace/dflash_sgl",DEV); taps=list(dcfg.target_layer_ids)
def peak(tag):
    print(f"  {tag:28s} peak={torch.cuda.max_memory_allocated()/2**30:6.2f} GiB  now={torch.cuda.memory_allocated()/2**30:6.2f}")
    torch.cuda.reset_peak_memory_stats()
peak("after models")
a=max(anchors,key=lambda a: seqs[a["prompt_id"]]["prompt_len"]+a["t"])
s=seqs[a["prompt_id"]]; full=s["prompt_ids"]+s["response_ids"]; cut=s["prompt_len"]+a["t"]
print("longest anchor context =",cut)
prefix=torch.tensor([full[:cut]],device=DEV)
with torch.no_grad():
    th=target(input_ids=prefix,output_hidden_states=True,use_cache=False)
    peak("target hidden-states fwd")
    print("    logits shape",tuple(th.logits.shape))
    thid=extract_context_feature(th.hidden_states,taps).to(torch.bfloat16)
    del th
    peak("extract_context_feature")
    q=drafter_q(draft,dcfg,thid,prefix,K,DEV); peak("drafter_q")
    gen=torch.Generator(device=DEV); gen.manual_seed(1)
    slots,chunk=rollout(target,prefix,256,K,C.CORPORA["C0"],stop_ids,gen,3*(1<<30))
    peak(f"rollout (chunk={chunk})")
    for k in range(K):
        T=mean_tv(slots[k],barycentre(slots[k])); R=mean_tv(slots[k],q[k])
    peak("barycentre+TV")

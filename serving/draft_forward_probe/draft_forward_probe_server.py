"""Launch the installed SGLang with a process-local, fixed-state draft timing probe.

No installed serving code is modified. A real request supplies initialized KV/cache
metadata. At the first selected batch at each exact prefix length, repeatedly
run that *same* proposal without advancing verification or committed prefix state.
T=0 only. CUDA events separately bracket ModelRunner.forward and the full proposal.
The model-forward graph may include a folded head: it is NOT called backbone-only.
"""
import json
import os
from pathlib import Path
import statistics
import sys

import torch
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer


ORIGINAL = DraftBlockProposer.propose
SEEN = set()
CONTEXTS = tuple(map(int, os.environ.get("DRAFT_PROBE_CONTEXTS", "512,1024").split(",")))
BATCHES = tuple(map(int, os.environ.get("DRAFT_PROBE_BATCHES", "1,32").split(",")))


def summary(values):
    v = sorted(values)
    return dict(n=len(v), mean_ms=statistics.mean(v), median_ms=statistics.median(v),
                min_ms=v[0], max_ms=v[-1], samples_ms=values)


def probe(self, **kwargs):
    bs = kwargs["bs"]
    if bs not in BATCHES:
        return ORIGINAL(self, **kwargs)
    lens = kwargs["batch"].seq_lens_cpu
    if lens is None:
        # The spec scheduler may invalidate its CPU mirror. This one metadata
        # read happens BEFORE warmup/timing; never inside either timed interval.
        lens = kwargs["batch"].seq_lens.cpu()
    lens_list = lens.tolist()
    lo, hi = min(lens_list), max(lens_list)
    context = lo if lo == hi and lo in CONTEXTS else None
    if not hasattr(self, "_probe_reported_shapes"):
        self._probe_reported_shapes = set()
    if (bs, lo, hi) not in self._probe_reported_shapes:
        print(f"DRAFT_PROBE_SHAPE bs={bs} prefix={lo}..{hi}", flush=True)
        self._probe_reported_shapes.add((bs, lo, hi))
    key = (bs, context)
    if context is None or key in SEEN:
        return ORIGINAL(self, **kwargs)
    if kwargs["sampling_info"] is not None and not kwargs["sampling_info"].is_all_greedy:
        raise RuntimeError("fixed-state probe requires greedy sampling")
    SEEN.add(key)
    runner = self.draft_model_runner
    forward = runner.forward
    # Untimed repeated writes touch only the same speculative slots; no verifier or
    # commit_hidden call advances the prefix between these replays.
    with torch.inference_mode():
        for _ in range(12):
            result = ORIGINAL(self, **kwargs)
        reference_tokens = result.draft_block.draft_tokens.clone()
        torch.cuda.synchronize()
        count = 64
        outer = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(count)]
        inner = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                 for _ in range(count)]
        graph_flags = []
        index = 0

        def timed_forward(*args, **kw):
            start, end = inner[index]
            start.record()
            result = forward(*args, **kw)
            end.record()
            graph_flags.append(bool(result.can_run_graph))
            return result

        runner.forward = timed_forward
        try:
            for index in range(count):
                start, end = outer[index]
                start.record()
                result = ORIGINAL(self, **kwargs)
                end.record()
            torch.cuda.synchronize()
        finally:
            runner.forward = forward
    assert len(graph_flags) == count, "expected one draft forward per proposal"
    assert all(graph_flags) and result.folded, "probe requires the folded CUDA graph path"
    assert torch.equal(reference_tokens, result.draft_block.draft_tokens), "fixed-state output changed"
    record = dict(
        arm=os.environ["DRAFT_PROBE_ARM"], batch_size=bs, context_label=context,
        prefix_lengths=lens_list, gamma=self.gamma, query_tokens=self.query_token_num,
        head_type=getattr(self.draft_model.markov_head, "markov_head_type", "vanilla"),
        lattice_top_k=self._lattice_top_k, folded=bool(result.folded),
        graph_replay_count=sum(graph_flags), pid=os.getpid(),
        fixed_state_output_stable=True, warmup_iterations=12,
        device=torch.cuda.get_device_name(),
        serving_module=sys.modules[DraftBlockProposer.__module__].__file__,
        draft_model_forward=summary([s.elapsed_time(e) for s,e in inner]),
        full_proposal=summary([s.elapsed_time(e) for s,e in outer]),
        scope="actual serving fixed-state replay; excludes target verify and commit_hidden",
    )
    path = Path(os.environ["DRAFT_PROBE_OUT"])
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print("DRAFT_FORWARD_PROBE " + json.dumps({k:v for k,v in record.items()
          if k not in ("draft_model_forward", "full_proposal", "prefix_lengths")}) +
          f" forward={record['draft_model_forward']['median_ms']:.4f}ms"
          f" proposal={record['full_proposal']['median_ms']:.4f}ms", flush=True)
    return result


DraftBlockProposer.propose = probe
print(f"DRAFT_PROBE_INSTALLED pid={os.getpid()}", flush=True)

if __name__ == "__main__":
    from sglang.launch_server import run_server, load_plugins, prepare_server_args, kill_process_tree
    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    SampleStepTokens,
)
from sglang.srt.environ import envs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.dspark_components.dspark_block_accept_estimator import (
    block_accept_estimate_enabled,
)
from sglang.srt.speculative.dspark_components.dspark_planner import VerifyWindow
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    spec_scale_global_num_tokens,
)
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils.invariants import Bucket, Invariant, NotNaN, expect

logger = logging.getLogger(__name__)


def _one_hot_token0(probs: torch.Tensor) -> torch.Tensor:
    degenerate = torch.isnan(probs[:, :1])
    one_hot = torch.zeros_like(probs)
    one_hot[:, 0] = 1.0
    return torch.where(degenerate, one_hot, probs)


# Draft step logits: NaN is a bug, but -inf is legitimate (masking). The data
# layer lives downstream (probs one-hot below, or the fast kernel's clamp).
_DRAFT_STEP_LOGITS = Invariant("dspark.draft.step_logits", Bucket.GUARD, NotNaN())
# Draft sampling probs: SOFTEN (tolerate + count), matching the original
# unconditional clamp; an all-NaN row would otherwise make multinomial raise.
_DRAFT_PROBS = Invariant(
    "dspark.draft.probs", Bucket.SOFTEN, NotNaN(), recover=_one_hot_token0
)


def _make_num_token_non_padded(
    num_tokens: int, device: str | torch.device
) -> Optional[torch.Tensor]:
    if not enable_num_token_non_padded():
        return None
    return torch.tensor(num_tokens, dtype=torch.int32).to(device, non_blocking=True)


class DraftBlockResult(msgspec.Struct, frozen=True):
    draft_tokens: torch.Tensor
    corrected_logits: Optional[torch.Tensor]
    greedy_mask: torch.Tensor
    temperatures: torch.Tensor
    # The sampling accept path wants the draft's distribution. A SEQUENTIAL proposal
    # supplies it as logits (`corrected_logits`) and lets SoftmaxTemp turn them into
    # probabilities. A LATTICE proposal already has probabilities -- the walk applied
    # the temperature to produce them -- so it fills this instead and the accept path
    # uses it verbatim. Sending a lattice q through the logits field meant log() on the
    # way in and a second division by the temperature on the way out, i.e. the verifier
    # scored against softmax(scores/T^2); see dflash_worker_v2's own handling, which
    # never round-trips through logits for exactly this reason.
    draft_probs: Optional[torch.Tensor] = None


class DraftForwardResult(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    raw_hidden: torch.Tensor
    draft_hidden_3d: torch.Tensor
    can_run_graph: bool


class DraftProposal(msgspec.Struct, frozen=True):
    draft_block_ids: torch.Tensor
    draft_block: DraftBlockResult
    draft_hidden: Optional[torch.Tensor]
    confidence: Optional[torch.Tensor] = None
    confidence_tap: Optional[torch.Tensor] = None
    folded: bool = False


def select_draft_hidden_without_anchor(
    hidden_states: torch.Tensor,
    *,
    bs: int,
    gamma: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_token_num = gamma + 1
    expected_rows = bs * query_token_num
    if hidden_states.shape[0] != expected_rows:
        raise RuntimeError(
            f"DSpark draft returned {hidden_states.shape[0]} hidden rows, "
            f"expected {expected_rows}."
        )
    hidden_by_query = hidden_states.view(bs, query_token_num, *hidden_states.shape[1:])
    selected = hidden_by_query[:, 1:].contiguous()
    return (
        selected.view(bs * gamma, *hidden_states.shape[1:]),
        selected.view(bs, gamma, -1),
    )


def make_next_draft_input(
    *,
    bonus_tokens: torch.Tensor,
    new_seq_lens: torch.Tensor,
) -> DFlashDraftInputV2:
    return make_draft_input_v2(bonus_tokens=bonus_tokens, new_seq_lens=new_seq_lens)


def resolve_greedy_mask(
    *,
    bs: int,
    sampling_info,
    device: torch.device,
) -> torch.Tensor:
    if sampling_info is None:
        return torch.ones(bs, dtype=torch.bool, device=device)
    return (sampling_info.top_ks <= 1).view(-1)


def _propose_by_lattice(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    markov_head,
    head_context,
    top_k: int,
    greedy_mask: torch.Tensor,
    temperatures: torch.Tensor,
    any_sampling: bool,
    device: torch.device,
    probs_sink=None,
) -> DraftBlockResult:
    """Propose over the candidate lattice the nomination term trained u to nominate.

    Top_k(u) per slot, every (slot, predecessor) edge scored in one batched call, then
    one walk over the precomputed [bs, gamma, K, K] table -- the DFlash2 selector's own
    machinery, reached through build_markov_lattice.
    """
    from sglang.kernels.ops.speculative.dflash import selector_walk_triton
    from sglang.srt.models.dspark import build_markov_lattice

    unary_logits, candidate_ids = base_logits.topk(top_k, dim=-1)
    scores = build_markov_lattice(
        markov_head,
        candidate_ids=candidate_ids,
        unary_logits=unary_logits.float(),
        hidden_states=draft_hidden,
        anchor_token_ids=anchor_tokens,
        context=head_context,
    )
    bs, slots = candidate_ids.shape[0], candidate_ids.shape[1]
    tokens, q_rows = selector_walk_triton(
        candidate_ids=candidate_ids,
        scores=scores.float(),
        uniforms=torch.rand(bs, slots, device=device),
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )

    # Under a lattice proposal the draft's distribution is supported on the candidates
    # and nowhere else, so the accept path is handed those probabilities scattered into a
    # dense row -- NOT a logits field it would re-temperature. `probs_sink` owns the
    # buffer because it must outlive this call: the accept happens in a later step.
    draft_probs = None
    if any_sampling and probs_sink is not None:
        draft_probs = probs_sink(candidate_ids, q_rows)
    return DraftBlockResult(
        draft_tokens=tokens,
        corrected_logits=None,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
        draft_probs=draft_probs,
    )


def sample_draft_block(
    *,
    base_logits: torch.Tensor,
    anchor_tokens: torch.Tensor,
    draft_hidden: torch.Tensor,
    sampling_info,
    markov_head,
    device: torch.device,
    head_context=None,
    lattice_top_k: int = 0,
    probs_sink=None,
) -> DraftBlockResult:
    bs = base_logits.shape[0]
    greedy_mask = resolve_greedy_mask(bs=bs, sampling_info=sampling_info, device=device)
    any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
    fast_sampling = envs.SGLANG_DSPARK_FAST_SAMPLING.get()

    if sampling_info is None:
        temperatures = torch.ones(bs, dtype=torch.float32, device=device)
    else:
        temperatures = (
            sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
        )

    if lattice_top_k > 0:
        return _propose_by_lattice(
            probs_sink=probs_sink,
            base_logits=base_logits,
            anchor_tokens=anchor_tokens,
            draft_hidden=draft_hidden,
            markov_head=markov_head,
            head_context=head_context,
            top_k=lattice_top_k,
            greedy_mask=greedy_mask,
            temperatures=temperatures,
            any_sampling=any_sampling,
            device=device,
        )

    if not any_sampling:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            expect(_DRAFT_STEP_LOGITS, step_logits, msg=f"step {step_idx}")
            return torch.argmax(step_logits, dim=-1)

    else:

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            expect(_DRAFT_STEP_LOGITS, step_logits, msg=f"step {step_idx}")
            if fast_sampling:
                exp_noise = torch.empty(
                    step_logits.shape, dtype=torch.float32, device=step_logits.device
                ).exponential_(1)
                return SampleStepTokens.execute(
                    step_logits=step_logits,
                    temperatures=temperatures,
                    greedy_mask=greedy_mask,
                    exp_noise=exp_noise,
                )
            else:
                probs = torch.softmax(
                    step_logits.float() / temperatures[:, None], dim=-1
                )
                probs = expect(_DRAFT_PROBS, probs)
                argmax_tokens = torch.argmax(step_logits, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                return torch.where(greedy_mask, argmax_tokens, sampled_tokens)

    # Only the attention head takes a context; passing it unconditionally would break
    # every table-shaped head's signature.
    context_kwargs = {} if head_context is None else {"context": head_context}
    # Greedy accept compares TOKENS, so the block logits are dead weight unless
    # something is sampling or the block-accept estimator is recording.  Same predicate
    # the recorder is built from, so the two cannot drift apart.
    draft_tokens, corrected_logits = markov_head.sample_block(
        base_logits,
        first_prev_tokens=anchor_tokens,
        hidden_states=draft_hidden,
        sampler=sampler,
        collect_logits=bool(any_sampling) or block_accept_estimate_enabled(),
        **context_kwargs,
    )
    return DraftBlockResult(
        draft_tokens=draft_tokens,
        corrected_logits=corrected_logits,
        greedy_mask=greedy_mask,
        temperatures=temperatures,
    )


class DraftBlockProposer:
    def __init__(
        self,
        *,
        draft_model,
        draft_model_runner,
        gamma: int,
        mask_token_id: int,
        draft_block_spec_info,
        lattice_top_k: int = 0,
        dp_moe_sync: bool = False,
    ) -> None:
        self.draft_model = draft_model
        self.draft_model_runner = draft_model_runner
        self.gamma = gamma
        self.sample_from_anchor = bool(draft_model.sample_from_anchor)
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self._mask_token_id = mask_token_id
        self._draft_block_spec_info = draft_block_spec_info
        # > 0 when the checkpoint's nomination term trained the unary head to nominate
        # a candidate set of this width; the proposal is a walk over that lattice
        # rather than a full-vocabulary argmax chain.
        self._lattice_top_k = int(lattice_top_k)
        self._draft_sampler = None
        self._dp_moe_sync = dp_moe_sync
        # Dense q for the sampling accept path, kept across rounds. See
        # `_lattice_draft_probs`; None until the first sampling batch.
        self._draft_probs_buf = None
        self._draft_probs_dirty = None

    def attach_draft_sampler(self, draft_sampler) -> None:
        self._draft_sampler = draft_sampler

    def _lattice_draft_probs(self, candidate_ids, q_rows):
        """A dense q carrying only the lattice's top_k per row, without paying for it.

        This is dflash_worker_v2's construction (see `_verify_sampling` there), and the
        reason is its own comment: "A fresh dense q would zero the whole vocabulary to
        carry top_k per row."  Allocating and filling [bs, gamma, 151936] every round is
        4.25 MB of writes and two extra dispatches at bs=1, in a loop that is dispatch
        bound -- measured as a uniform +1.2~1.6% per round across all nine datasets,
        which is most of what this arm's accept-length advantage buys back.

        So the buffer persists and only the entries actually written are cleared. The
        clear happens HERE, against the PREVIOUS round's ids, rather than after the
        accept: proposal and accept are separate calls in this worker, and the ids are a
        static graph buffer the next replay overwrites -- hence the clone. Leaving stale
        rows would hand the verifier probability mass on tokens this round never
        proposed, which is silent: acceptance would just be subtly wrong.
        """
        bs, gamma, _ = candidate_ids.shape
        vocab = int(self.draft_model.lm_head.org_vocab_size)
        buf = self._draft_probs_buf
        if buf is None or buf.shape[0] < bs or tuple(buf.shape[1:]) != (gamma, vocab):
            cap = bs if buf is None else max(bs, buf.shape[0] * 2)
            buf = torch.zeros(
                (cap, gamma, vocab), dtype=torch.float32, device=candidate_ids.device
            )
            self._draft_probs_buf = buf
            self._draft_probs_dirty = None
        if self._draft_probs_dirty is not None:
            prev_ids, prev_bs = self._draft_probs_dirty
            buf[:prev_bs].scatter_(-1, prev_ids, 0.0)
        probs = buf[:bs]
        probs.scatter_(-1, candidate_ids, q_rows.float())
        self._draft_probs_dirty = (candidate_ids.clone(), bs)
        return probs

    def _base_logits_context(self):
        if self._dp_moe_sync:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def propose(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        target_model,
        sampling_info,
    ) -> DraftProposal:
        embed_module = unwrap_lora_layer(
            self.draft_model.embed_tokens
            if not self.sample_from_anchor
            else target_model.get_input_embeddings()
        )
        draft_sampler = self._draft_sampler
        all_greedy = sampling_info is None or sampling_info.is_all_greedy
        # The attention head reads the committed prefix, so it cannot run inside the
        # fused folded-proposal sampler -- that kernel walks the block from
        # (prev_token, hidden) alone.  Disabling the fast path here beats letting it
        # produce a block the head never corrected.
        head_reads_prefix = (
            getattr(self.draft_model.markov_head, "markov_head_type", "") == "attn"
        )
        fwd = self._run_forward(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            embed_module=embed_module,
            draft_sampler=draft_sampler,
            sampling_info=sampling_info,
        )
        draft_block_ids = fwd.draft_block_ids

        folded_confidence = None
        confidence_tap = None
        folded = False
        if (
            envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get()
            and draft_sampler is not None
            # Two samplers can be attached, and they fold different proposals.
            # DsparkDraftSampler walks the FULL vocabulary, so folding it for a lattice
            # arm would silently bypass the candidate set that arm was trained for, and
            # its kernel walks from (prev_token, hidden) alone so it cannot run a head
            # that reads the committed prefix.  LatticeDraftSampler folds the lattice
            # itself and stages the head's prefix inputs, so it has neither limit --
            # which is the whole point: at bs=1 the round is dispatch-bound, and an
            # unfolded proposal measured ~5.5 ms of draft against ~1.9 ms folded.
            and (
                getattr(draft_sampler, "proposes_lattice", False)
                == bool(self._lattice_top_k)
            )
            and (
                getattr(draft_sampler, "proposes_lattice", False)
                or not head_reads_prefix
            )
            and fwd.can_run_graph
            # A LATTICE sampler folds sampling too, it just cannot hand the accept path a
            # vocabulary-wide q: that buffer is 1.09 GB static at max_bs=256, which is why
            # `folded_sampling` is False on it.  But a lattice q is supported on top_k
            # candidates and those ARE static buffers, so the block can be scattered out
            # after replay for the price of one kernel.  Leaving sampling batches eager
            # instead cost ~2.3 ms/round, measured: at temperature 1 this arm ran 755.8
            # tok/s against 1073.6 at temperature 0, while the sequential arm -- which
            # folds both -- did not move (1009.7 vs 1012.2).  That is an implementation
            # gap being read as an architecture difference.
            and (
                all_greedy
                or draft_sampler.folded_sampling
                or getattr(draft_sampler, "proposes_lattice", False)
            )
        ):
            folded = True
            lattice_probs = None
            if getattr(draft_sampler, "proposes_lattice", False):
                greedy_mask = draft_sampler.greedy_mask[:bs]
                temperatures = draft_sampler.temperatures[:bs]
                corrected_logits = None
                if not all_greedy:
                    # Same construction as the eager path: probabilities into the
                    # persistent buffer, never logits. See _lattice_draft_probs.
                    lattice_probs = self._lattice_draft_probs(
                        draft_sampler.candidate_out[:bs], draft_sampler.q_out[:bs]
                    )
            elif draft_sampler.folded_sampling:
                greedy_mask = draft_sampler.greedy_mask[:bs]
                temperatures = draft_sampler.temperatures[:bs]
                # The sampling accept path needs the markov-corrected block
                # logits; greedy accept only compares tokens.
                corrected_logits = (
                    None
                    if all_greedy
                    else draft_sampler.corrected_out[: bs * self.gamma].view(
                        bs, self.gamma, -1
                    )
                )
            else:
                # Greedy-only folding: the hook argmaxed every row and kept no
                # sampling buffers, so derive the params on the fly.
                greedy_mask = resolve_greedy_mask(
                    bs=bs, sampling_info=sampling_info, device=device
                )
                if sampling_info is None:
                    temperatures = torch.ones(bs, dtype=torch.float32, device=device)
                else:
                    temperatures = (
                        sampling_info.temperatures.view(-1)
                        .to(torch.float32)
                        .clamp_min(1e-5)
                    )
                corrected_logits = None
            draft_block = DraftBlockResult(
                draft_tokens=draft_sampler.out[: bs * self.gamma].view(bs, self.gamma),
                corrected_logits=corrected_logits,
                greedy_mask=greedy_mask,
                temperatures=temperatures,
                draft_probs=lattice_probs,
            )
            if draft_sampler.confidence_out is not None:
                folded_confidence = draft_sampler.confidence_out[:bs]
        else:
            with self._base_logits_context():
                base_logits, confidence_tap = self.draft_model.compute_base_logits(
                    fwd.raw_hidden
                )
                base_logits = base_logits.view(bs, self.gamma, -1)
            head_context = None
            if head_reads_prefix:
                # cache_seqlens is batch.seq_lens, NOT seq_lens + query_token_num: the
                # anchor sits at position seq_len and the head must read [0, seq_len)
                # only.  The next position holds the target's POST-anchor
                # representation, which decodes to the slot-1 answer.
                # Slice the COLUMNS before gathering the rows.  req_to_token rows are
                # max_context_len wide (the pool's, not this batch's), so indexing the
                # rows first materialises [bs, max_context_len] every decode round and
                # throws almost all of it away -- tens of MB per round at long context
                # or large batch.  The head only ever reads [0, seq_len] plus the anchor
                # slot at seq_len, so max(seq_lens) + 1 columns is the whole requirement.
                #
                # The width comes off seq_lens_cpu so it costs no GPU-to-CPU sync; the
                # fallback path is only for batches that carry no CPU copy.
                if batch.seq_lens_cpu is not None:
                    width = int(batch.seq_lens_cpu.max()) + 1
                else:
                    width = int(batch.seq_lens.max().item()) + 1
                page_table = self.draft_model_runner.req_to_token_pool.req_to_token[
                    batch.req_pool_indices, :width
                ]
                head_context = self.draft_model.markov_head.build_serving_context(
                    page_table=page_table,
                    prefix_lens=batch.seq_lens,
                    anchor_token_ids=draft_block_ids[:, 0],
                    # Slot 0 of the verify window IS position prefix_len, which is where
                    # the anchor lives.  Passed explicitly rather than read back out of
                    # req_to_token, which may not have been updated for this window yet.
                    anchor_cache_loc=verify_window.verify_cache_loc_2d[:, 0],
                )
            draft_block = sample_draft_block(
                probs_sink=self._lattice_draft_probs,
                base_logits=base_logits,
                anchor_tokens=draft_block_ids[:, 0],
                draft_hidden=fwd.draft_hidden_3d,
                sampling_info=sampling_info,
                markov_head=self.draft_model.markov_head,
                device=device,
                head_context=head_context,
                lattice_top_k=self._lattice_top_k,
            )
        proposal_block_ids = (
            draft_block_ids
            if self.sample_from_anchor
            else draft_block_ids[:, : self.gamma].contiguous()
        )
        return DraftProposal(
            draft_block_ids=proposal_block_ids,
            draft_block=draft_block,
            draft_hidden=fwd.draft_hidden_3d,
            confidence=folded_confidence,
            confidence_tap=confidence_tap,
            folded=folded,
        )

    def run_idle_participation(self, batch: ScheduleBatch) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        device = self.draft_model_runner.device
        empty_long = torch.empty((0,), dtype=torch.int64, device=device)
        idle_batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=empty_long,
            req_pool_indices=empty_long,
            seq_lens=empty_long,
            out_cache_loc=empty_long,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int64),
            positions=empty_long,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._fill_dp_moe_sync_metadata(idle_batch, batch)
        with torch.inference_mode():
            self.draft_model_runner.forward(idle_batch)

    def _run_forward(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        verify_window: VerifyWindow,
        bs: int,
        device: str,
        embed_module,
        draft_sampler=None,
        sampling_info=None,
    ) -> DraftForwardResult:
        gamma = self.gamma
        query_token_num = self.query_token_num
        prefix_lens = batch.seq_lens
        positions_2d = verify_window.positions_2d
        verify_cache_loc_2d = verify_window.verify_cache_loc_2d

        draft_block_ids = torch.full(
            (bs, query_token_num),
            int(self._mask_token_id),
            dtype=torch.long,
            device=device,
        )
        draft_block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        draft_positions = positions_2d[:, :query_token_num].reshape(-1)
        draft_cache_loc = verify_cache_loc_2d[:, :query_token_num].reshape(-1)

        draft_owns_embed = envs.SGLANG_DSPARK_EMBED_IN_GRAPH.get() and hasattr(
            self.draft_model, "forward_embed"
        )
        draft_input_embeds: Optional[torch.Tensor] = None
        if not draft_owns_embed:
            noise_embedding = embed_module(draft_block_ids)
            draft_input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu = batch.seq_lens_cpu + query_token_num
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        elif draft_input.nxt_kv_lens_cpu is not None:
            draft_seq_lens_cpu = draft_input.nxt_kv_lens_cpu
            draft_seq_lens_sum = int(draft_input.nxt_kv_lens_sum)
        else:
            raise RuntimeError("DSpark decode expected batch.seq_lens_cpu, got None")

        draft_num_tokens = bs * query_token_num
        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=draft_block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=draft_positions,
            input_embeds=draft_input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            num_token_non_padded=_make_num_token_non_padded(draft_num_tokens, device),
            num_token_non_padded_cpu=draft_num_tokens,
        )
        self._fill_dp_moe_sync_metadata(draft_forward_batch, batch)
        graph_runner = self.draft_model_runner.decode_cuda_graph_runner
        if (
            draft_sampler is not None
            and graph_runner is not None
            and graph_runner.can_run_graph(draft_forward_batch)
        ):
            draft_sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
            if hasattr(draft_sampler, "stage_head_context"):
                # The lattice sampler's head reads the committed prefix, and a captured
                # graph has those addresses baked in -- so the page table and cache
                # lengths have to be written into buffers that never move, HERE, before
                # the replay consumes them.  The eager path instead rebuilds a fresh
                # tensor inside the proposal every round.
                draft_sampler.stage_head_context(
                    bs=bs,
                    req_to_token=self.draft_model_runner.req_to_token_pool.req_to_token,
                    req_pool_indices=batch.req_pool_indices,
                    seq_lens=batch.seq_lens,
                    seq_lens_cpu=batch.seq_lens_cpu,
                    anchor_token_ids=draft_block_ids[:, 0],
                    anchor_cache_loc=verify_cache_loc_2d[:, 0],
                    # Host-side upper bound for the page-table width, so the staging
                    # never has to read seq_lens back off the GPU. See the branch it
                    # feeds in stage_head_context.
                    nxt_kv_lens_cpu=draft_input.nxt_kv_lens_cpu,
                )
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        logits_output = draft_out.logits_output
        raw_hidden = logits_output.hidden_states
        if raw_hidden is None:
            raise RuntimeError("DSpark draft model returned no hidden states.")
        if self.sample_from_anchor:
            expected_rows = bs * gamma
            if raw_hidden.shape[0] != expected_rows:
                raise RuntimeError(
                    f"DSpark draft returned {raw_hidden.shape[0]} hidden rows, "
                    f"expected {expected_rows}."
                )
            model_hidden = raw_hidden
            draft_hidden_3d = raw_hidden.view(bs, gamma, -1)
        else:
            model_hidden, draft_hidden_3d = select_draft_hidden_without_anchor(
                raw_hidden,
                bs=bs,
                gamma=gamma,
            )
        return DraftForwardResult(
            draft_block_ids=draft_block_ids,
            raw_hidden=model_hidden,
            draft_hidden_3d=draft_hidden_3d,
            can_run_graph=draft_out.can_run_graph,
        )

    def _fill_dp_moe_sync_metadata(
        self, forward_batch: ForwardBatch, batch: ScheduleBatch
    ) -> None:
        # The dense DSpark draft still reuses the target batch's graph tier.
        # Set graph eligibility before the DP-MoE-only metadata early return.
        forward_batch.can_run_decode_cuda_graph = batch.can_run_decode_cuda_graph
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
        # Graph bucket selection uses the raw per-rank request counts.  Keep
        # them separate from global_num_tokens_cpu below, which is scaled into
        # draft-token units for DP/MoE synchronization.
        forward_batch.original_global_num_tokens_cpu = batch.global_num_tokens
        gnt, gnt_logprob = spec_scale_global_num_tokens(
            self._draft_block_spec_info,
            batch.global_num_tokens,
            batch.global_num_tokens_for_logprob,
        )
        device = self.draft_model_runner.device
        forward_batch.original_global_num_tokens_cpu = batch.global_num_tokens
        num_tokens = forward_batch.input_ids.numel()
        if enable_num_token_non_padded():
            forward_batch.num_token_non_padded = torch.tensor(
                num_tokens, dtype=torch.int32, device=device
            )
        forward_batch.num_token_non_padded_cpu = num_tokens
        forward_batch.global_num_tokens_cpu = gnt
        forward_batch.global_num_tokens_for_logprob_cpu = gnt_logprob
        forward_batch.global_num_tokens_gpu = torch.tensor(gnt, dtype=torch.int64).to(
            device, non_blocking=True
        )
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            gnt_logprob, dtype=torch.int64
        ).to(device, non_blocking=True)

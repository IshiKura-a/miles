"""FSDP learner policy for SLMv3 Stage5-style reinforcement learning."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import torch
import torch.distributed as dist
from miles.utils.distributed_utils import get_gloo_group
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.t5.modeling_t5 import T5Attention
from examples.experimental.generative_retriever.model.constrained_decode import (
    PrefixConstraint,
    SortedU32Constraint,
    load_constraint,
)
from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping
from examples.experimental.generative_retriever.sglang_models.gr_policy import _stage_local_mapping_dir
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    DOC2K_ADD_MIN_QUERIES,
    DOC2K_PRUNE_FLOOR,
    DOC2K_PRUNE_MIN_QUERIES,
)

from examples.experimental.generative_retriever.model.model_loading import (
    build_run_arguments,
    checkpoint_tensor_shape,
    import_slmv3,
    load_checkpoint,
    require_directory,
    stage_local_model_dir,
)


def _required_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"{variable} must be set for the GR FSDP model factory")
    return stage_local_model_dir(value, variable)


class GRStage5Policy(nn.Module):
    """Trainable Qwen encoder, projection, and autoregressive docid decoder."""

    _miles_requires_response_lengths = True
    _miles_fsdp_ignore_frozen_parameters = True
    _miles_fsdp_wrap_root_only = True
    _no_split_modules = ["Qwen3DecoderLayer", "T5Block"]

    def __init__(
        self, source_model: nn.Module, pad_token_id: int, constraint: PrefixConstraint,
    ) -> None:
        super().__init__()
        source: Any = source_model
        self.config = source.encoder_list[0].model.encoder.config
        self.config.tie_word_embeddings = False
        self.pad_token_id = pad_token_id
        self.code_dim: int = source.docid_dim
        self.code_num: int = source.docid_size
        self.constraint = constraint
        self._prefix_allowed_cache: dict[tuple[int, ...], tuple[torch.Tensor, set[int]]] = {}
        self._global_cand_votes: dict[tuple[int, int], int] = {}
        self._global_docid_q_count: dict[int, int] = {}
        self._global_docid_kids: dict[int, tuple[int, ...]] = {}
        self._global_edge_passes: dict[tuple[int, int], int] = {}
        object.__setattr__(self, "_doc2k_sync_mapping", None)

        self.encoder_list: Any = source.encoder_list
        self.docEnc2decoder: Any = source.docEnc2decoder
        self.decoder: Any = source.decoder
        self.lm_head: Any = source.lm_head

        causal_lm = self.encoder_list[0].model.encoder
        if hasattr(causal_lm, "lm_head"):
            causal_lm.lm_head = nn.Identity()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, T5Attention):
                module.dropout = 0.0
        for module in (
            self.encoder_list,
            self.docEnc2decoder,
            self.decoder,
            self.lm_head,
        ):
            module.requires_grad_(True)

    def _encoder_backbone(self) -> Any:
        causal_lm = self.encoder_list[0].model.encoder
        backbone = getattr(causal_lm, "model", None)
        if backbone is None:
            raise TypeError("GR Qwen encoder must expose its backbone as .model")
        return backbone

    def _sync_global_doc2k_votes(self) -> None:
        if not dist.is_initialized():
            return
        gloo_group = get_gloo_group()
        rank = dist.get_rank()
        local_gpu = rank % 8
        if rank == 0:
            self._ensure_doc2k_sync_mapping()
        local_path = Path(tempfile.gettempdir()) / f"gr_doc2k_local_gpu{local_gpu}.pkl"
        local_payload: dict[str, Any] | None = None
        if local_path.is_file():
            try:
                local_payload = pickle.loads(local_path.read_bytes())
            finally:
                local_path.unlink(missing_ok=True)
        world_size = dist.get_world_size()
        gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0, group=gloo_group)

        delta_box: list[dict[str, Any] | None] = [None]
        if rank == 0 and gathered is not None and any(item is not None for item in gathered):
            cand_votes = self._global_cand_votes
            docid_q_count = self._global_docid_q_count
            edge_passes = self._global_edge_passes
            sync_map = self._ensure_doc2k_sync_mapping()

            for item in gathered:
                if item is None:
                    continue
                for ckey, cnt in item["cand_votes"].items():
                    cand_votes[ckey] = cand_votes.get(ckey, 0) + int(cnt)
                for u32, qcnt in item["docid_q_count"].items():
                    docid_q_count[u32] = docid_q_count.get(u32, 0) + int(qcnt)
                for ekey, pcnt in item["edge_passes"].items():
                    edge_passes[ekey] = edge_passes.get(ekey, 0) + int(pcnt)

            additions: dict[int, list[int]] = {}
            promoted_ckeys: list[tuple[int, int]] = []
            for ckey, total_votes in sorted(cand_votes.items()):
                if total_votes >= DOC2K_ADD_MIN_QUERIES:
                    target_u32, kid = ckey
                    bucket_list = additions.setdefault(target_u32, [])
                    if kid not in bucket_list:
                        bucket_list.append(kid)
                    promoted_ckeys.append(ckey)
            for ckey in promoted_ckeys:
                cand_votes.pop(ckey, None)

            removals: dict[int, set[int]] = {}
            harvested_u32: list[tuple[int, tuple[int, ...]]] = []
            for u32, total_q in sorted(docid_q_count.items()):
                if total_q < DOC2K_PRUNE_MIN_QUERIES:
                    continue
                kids_tup: tuple[int, ...] = ()
                if sync_map is not None:
                    ov_kids = sync_map._overlay_kw_ids.get(u32)
                    if ov_kids is not None:
                        kids_tup = ov_kids
                    else:
                        base_kids = sync_map._base_kw_ids_one(u32)
                        if base_kids is not None:
                            kids_tup = tuple(dict.fromkeys(base_kids))
                harvested_u32.append((u32, kids_tup))
                if len(kids_tup) <= DOC2K_PRUNE_FLOOR:
                    continue
                max_rem = max(0, len(kids_tup) - DOC2K_PRUNE_FLOOR)
                dead_kids: set[int] = set()
                for kid in kids_tup:
                    if edge_passes.get((u32, kid), 0) == 0:
                        dead_kids.add(kid)
                        if len(dead_kids) >= max_rem:
                            break
                if dead_kids:
                    removals[u32] = dead_kids
            for u32, kids_tup in harvested_u32:
                docid_q_count.pop(u32, None)
                for kid in kids_tup:
                    edge_passes.pop((u32, kid), None)

            if sync_map is not None:
                sync_map.apply_u32_kid_delta(additions, removals)
            delta_box[0] = {"additions": additions, "removals": removals}

        dist.broadcast_object_list(delta_box, src=0, group=gloo_group)
        delta = delta_box[0]
        if delta is None:
            return

        additions = delta["additions"]
        removals = delta["removals"]
        global_path = Path(tempfile.gettempdir()) / f"gr_doc2k_global_gpu{local_gpu}.pkl"
        if global_path.is_file():
            prev = pickle.loads(global_path.read_bytes())
            for u32, kids in prev.get("additions", {}).items():
                bl = additions.setdefault(u32, [])
                for k in kids:
                    if k not in bl:
                        bl.append(k)
            for u32, kids in prev.get("removals", {}).items():
                removals.setdefault(u32, set()).update(kids)

        tmp_path = global_path.with_suffix(".tmp")
        tmp_path.write_bytes(
            pickle.dumps({"additions": additions, "removals": removals}, protocol=pickle.HIGHEST_PROTOCOL)
        )
        tmp_path.replace(global_path)

    def _ensure_doc2k_sync_mapping(self) -> ResidentMapping | None:
        sync_map: ResidentMapping | None = getattr(self, "_doc2k_sync_mapping", None)
        if sync_map is not None:
            return sync_map
        raw_mapping = os.environ.get("GR_ID_MAPPING")
        if not raw_mapping:
            return None
        staged_dir = _stage_local_mapping_dir(raw_mapping)
        sync_map = ResidentMapping(staged_dir, read_only=True)
        object.__setattr__(self, "_doc2k_sync_mapping", sync_map)
        return sync_map

    def save_doc2k_mapping(
        self,
        output_dir: str | Path,
        *,
        stage_dir: str | Path | None = None,
        async_transfer: bool = True,
    ) -> None:
        self._sync_global_doc2k_votes()
        if not dist.is_initialized() or dist.get_rank() == 0:
            sync_map = self._ensure_doc2k_sync_mapping()
            if sync_map is not None:
                sync_map.save_docid_mappings(
                    output_dir,
                    stage_dir=stage_dir,
                    async_transfer=async_transfer,
                )
        if dist.is_initialized():
            dist.barrier(group=get_gloo_group())

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._sync_global_doc2k_votes()
        return super().state_dict(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        del kwargs
        self.decoder.gradient_checkpointing = False

    def _encode_prompts(
        self,
        sample_tokens: list[torch.Tensor],
        response_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        prompt_lengths = [
            tokens.shape[0] - response_length
            for tokens, response_length in zip(sample_tokens, response_lengths, strict=True)
        ]
        n = len(sample_tokens)
        g_size = DEFAULT_TRAIN_GROUP_SIZE
        is_grouped = (
            n >= g_size
            and n % g_size == 0
            and all(
                prompt_lengths[start] == prompt_lengths[i]
                for start in range(0, n, g_size)
                for i in range(start + 1, start + g_size)
            )
        )
        if is_grouped:
            group_prompts = [
                sample_tokens[start][: prompt_lengths[start]].flip(0)
                for start in range(0, n, g_size)
            ]
            padded_groups = pad_sequence(
                group_prompts,
                batch_first=True,
                padding_value=self.pad_token_id,
            ).flip(1)
            unique_prompts, group_inverse = torch.unique(
                padded_groups,
                dim=0,
                return_inverse=True,
            )
            inverse_indices = group_inverse.repeat_interleave(g_size)
        else:
            prompts = [
                sample_tokens[index][:prompt_length]
                for index, prompt_length in enumerate(prompt_lengths)
            ]
            padded_prompts = pad_sequence(
                [prompt.flip(0) for prompt in prompts],
                batch_first=True,
                padding_value=self.pad_token_id,
            ).flip(1)
            unique_prompts, inverse_indices = torch.unique(
                padded_prompts,
                dim=0,
                return_inverse=True,
            )
        unique_mask = unique_prompts.ne(self.pad_token_id).long()
        backbone = self._encoder_backbone()
        outputs = backbone(
            input_ids=unique_prompts,
            attention_mask=unique_mask,
            return_dict=True,
            use_cache=False,
        )
        unique_hidden_states = outputs.last_hidden_state
        return unique_hidden_states, unique_mask, inverse_indices, prompt_lengths

    def _allowed_for_prefix(self, prefix_tuple: tuple[int, ...]) -> tuple[torch.Tensor, set[int]]:
        cached = self._prefix_allowed_cache.get(prefix_tuple)
        if cached is not None:
            return cached
        allowed_list = self.constraint(0, torch.tensor(prefix_tuple, dtype=torch.long))
        allowed_tensor = torch.tensor(allowed_list, dtype=torch.long)
        entry = (allowed_tensor, set(allowed_list))
        if len(self._prefix_allowed_cache) < 262_144:
            self._prefix_allowed_cache[prefix_tuple] = entry
        return entry

    def _build_allowed_mask(
        self,
        responses: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        if isinstance(self.constraint, SortedU32Constraint):
            mask_gpu = self.constraint.build_response_mask(responses, vocab_size)
            if not bool(mask_gpu.gather(2, responses.unsqueeze(-1)).all()):
                raise ValueError("GR response violates official prefix constraint")
            return mask_gpu
        sample_count = responses.shape[0]
        responses_cpu: list[list[int]] = responses.detach().cpu().tolist()
        mask_cpu = torch.zeros((sample_count, self.code_num, vocab_size), dtype=torch.bool)
        for index, row in enumerate(responses_cpu):
            prefix: tuple[int, ...] = (0,)
            for position in range(self.code_num):
                allowed_tensor, allowed_set = self._allowed_for_prefix(prefix)
                token_id = row[position]
                if not allowed_set or token_id not in allowed_set:
                    raise ValueError(f"GR response violates official prefix constraint at position {position}")
                mask_cpu[index, position, allowed_tensor] = True
                prefix = (*prefix, token_id)
        return mask_cpu.to(device=responses.device, non_blocking=True)

    @staticmethod
    def _unpack_tokens(
        input_ids: torch.Tensor,
        total_lengths: list[int],
    ) -> tuple[list[torch.Tensor], list[int]]:
        if input_ids.ndim != 2:
            raise ValueError(f"GRStage5Policy expected rank-2 input_ids, got {input_ids.shape}")
        if input_ids.shape[0] == len(total_lengths):
            return (
                [
                    input_ids[index, :total_length]
                    for index, total_length in enumerate(total_lengths)
                ],
                [0] * len(total_lengths),
            )
        if input_ids.shape[0] != 1:
            raise ValueError(
                "GRStage5Policy requires Miles thd packed input or one padded row per sample"
            )

        samples = []
        offsets = []
        offset = 0
        for total_length in total_lengths:
            offsets.append(offset)
            samples.append(input_ids[0, offset : offset + total_length])
            offset += total_length
        return samples, offsets

    def forward(
        self,
        input_ids: torch.Tensor,
        response_lengths: list[int] | torch.Tensor,
        total_lengths: list[int] | torch.Tensor,
        **kwargs: Any,
    ) -> CausalLMOutput:
        del kwargs
        response_lengths = [int(length) for length in response_lengths]
        total_lengths = [int(length) for length in total_lengths]
        if any(length != self.code_num for length in response_lengths):
            raise ValueError(f"GR response lengths must all equal {self.code_num}")
        sample_tokens, output_offsets = self._unpack_tokens(input_ids, total_lengths)

        unique_hidden_states, unique_mask, inverse_indices, prompt_lengths = self._encode_prompts(
            sample_tokens,
            response_lengths,
        )
        unique_hidden_states = unique_hidden_states.to(self.docEnc2decoder.weight.dtype)
        unique_projected = self.docEnc2decoder(unique_hidden_states)
        projected_hidden_states = unique_projected.index_select(0, inverse_indices)
        attention_mask = unique_mask.index_select(0, inverse_indices)
        input_batch_size, sequence_length = input_ids.shape
        sample_count = len(sample_tokens)
        vocab_size = self.code_num * self.code_dim + 2
        logits = projected_hidden_states.new_zeros(
            (input_batch_size, sequence_length, vocab_size)
        )

        g_size = DEFAULT_TRAIN_GROUP_SIZE
        if (
            input_batch_size == 1
            and sample_count >= g_size
            and sample_count % g_size == 0
            and all(
                total_lengths[start] == total_lengths[i]
                for start in range(0, sample_count, g_size)
                for i in range(start + 1, start + g_size)
            )
        ):
            responses = torch.cat(
                [
                    input_ids[
                        0,
                        output_offsets[start] : output_offsets[start] + g_size * total_lengths[start],
                    ].view(g_size, total_lengths[start])[:, prompt_lengths[start] : prompt_lengths[start] + self.code_num]
                    for start in range(0, sample_count, g_size)
                ],
                dim=0,
            )
        else:
            responses = torch.stack([
                tokens[prompt_length : prompt_length + self.code_num]
                for tokens, prompt_length in zip(sample_tokens, prompt_lengths, strict=True)
            ])
        decoder_input_ids = torch.cat(
            (
                torch.zeros(
                    (sample_count, 1),
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                responses[:, :-1],
            ),
            dim=-1,
        )
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=projected_hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        response_logits = self.lm_head(decoder_outputs.last_hidden_state)
        invalid_logit = torch.finfo(response_logits.dtype).min
        allowed_mask = self._build_allowed_mask(responses, vocab_size)
        masked_response_logits = response_logits.masked_fill(~allowed_mask, invalid_logit)
        starts = torch.tensor(
            [
                output_offset + prompt_length - 1
                for prompt_length, output_offset in zip(prompt_lengths, output_offsets, strict=True)
            ],
            dtype=torch.long,
            device=input_ids.device,
        )
        positions = starts.unsqueeze(1) + torch.arange(self.code_num, device=input_ids.device).unsqueeze(0)
        if input_batch_size == 1:
            logits[0, positions.reshape(-1)] = masked_response_logits.reshape(-1, vocab_size)
        else:
            row_indices = torch.arange(sample_count, device=input_ids.device).unsqueeze(1).expand_as(positions)
            logits[row_indices, positions] = masked_response_logits
        return CausalLMOutput(logits=logits)


def build_gr_stage5_policy(checkpoint_path: str, args: Any, init_context) -> GRStage5Policy:
    """Miles ``--custom-fsdp-model-factory-path`` entrypoint."""
    del checkpoint_path, args

    checkpoint = _required_path("GR_CHECKPOINT")
    qwen_model = _required_path("GR_QWEN_MODEL")
    slm_root_value = os.environ.get("GR_SLM_ROOT")
    if not slm_root_value:
        raise ValueError("GR_SLM_ROOT must be set for the GR FSDP model factory")
    slm_root = require_directory(slm_root_value, "GR_SLM_ROOT")
    mapping = os.environ.get("GR_ID_MAPPING")
    if not mapping:
        raise ValueError("GR_ID_MAPPING must be set for constrained GR training")
    constraint = load_constraint(mapping, 4, 256)
    slmv3 = import_slmv3(slm_root)
    run_args = build_run_arguments(slmv3["RunArguments"], checkpoint, qwen_model)

    with init_context():
        t5_config = T5Config.from_pretrained(checkpoint)
        t5_config.vocab_size = checkpoint_tensor_shape(
            checkpoint,
            "query_encoder.embed_tokens.weight",
        )[0]
        t5_model = T5ForConditionalGeneration(t5_config)
        qwen_encoder = slmv3["Encoder_QWen1216t"](run_args)
        source_model = slmv3["T5MultiWithGivenQueryDocEncoderDSI"](
            t5_config,
            t5_model,
            encoder=None,
            docencoder=qwen_encoder,
            docid_dim=run_args.code_dim,
            docid_size=run_args.code_num,
            margin=run_args.margin,
            running_args=run_args,
            domain_idx=[0],
        )

    if not any(parameter.is_meta for parameter in source_model.parameters()):
        load_checkpoint(source_model, checkpoint)
    return GRStage5Policy(
        source_model,
        pad_token_id=qwen_encoder.tokenizer.pad_token_id,
        constraint=constraint,
    )

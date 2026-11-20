"""FSDP learner policy for SLMv3 Stage5-style reinforcement learning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.t5.modeling_t5 import T5Attention

from examples.experimental.generative_retriever.index.fast_doc2k_sync import FastDoc2kSyncState
from examples.experimental.generative_retriever.model.constrained_decode import (
    PrefixConstraint,
    SortedU32Constraint,
    install_grouped_cross_attention,
    load_constraint,
)
from examples.experimental.generative_retriever.model.model_loading import (
    build_run_arguments,
    checkpoint_tensor_shape,
    import_slmv3,
    load_checkpoint,
    require_directory,
    stage_local_model_dir,
)
from examples.experimental.generative_retriever.reward.in_process_protocol import (
    resolve_physical_gpu_id,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
)
from examples.experimental.generative_retriever.runtime.gr_config import GRConfig
from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping
from examples.experimental.generative_retriever.services.mapping_staging import (
    save_static_mapping_snapshot,
    stage_local_mapping_dir,
)
from miles.utils.distributed_utils import get_gloo_group

_INVALID_DOCID_LOGIT = -1.0e9


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
        self.gr_config = GRConfig.from_env()
        self._prefix_allowed_cache: dict[tuple[int, ...], tuple[torch.Tensor, set[int]]] = {}
        self._fast_doc2k_state = FastDoc2kSyncState()
        object.__setattr__(self, "_doc2k_sync_mapping", None)

        self.encoder_list: Any = source.encoder_list
        self.docEnc2decoder: Any = source.docEnc2decoder
        self.decoder: Any = source.decoder
        self.lm_head: Any = source.lm_head
        if self.gr_config.grouped_cross_attention:
            install_grouped_cross_attention(self.decoder)

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
        if self.gr_config.doc2k_disabled or not dist.is_initialized():
            return
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        local_gpu = resolve_physical_gpu_id(device, fallback_rank=rank)
        if rank == 0:
            self._ensure_doc2k_sync_mapping()
        self._fast_doc2k_state.sync_distributed(
            local_gpu=local_gpu,
            rank=rank,
            world_size=world_size,
            device=device,
            ensure_sync_map_fn=self._ensure_doc2k_sync_mapping,
        )

    def _ensure_doc2k_sync_mapping(self) -> ResidentMapping | None:
        sync_map: ResidentMapping | None = getattr(self, "_doc2k_sync_mapping", None)
        if sync_map is not None:
            return sync_map
        raw_mapping = os.environ.get("GR_ID_MAPPING")
        if not raw_mapping:
            return None
        staged_dir = stage_local_mapping_dir(raw_mapping)
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
        if self.gr_config.doc2k_disabled:
            if not dist.is_initialized() or dist.get_rank() == 0:
                raw_mapping = os.environ.get("GR_ID_MAPPING")
                if raw_mapping:
                    save_static_mapping_snapshot(
                        output_dir,
                        base_mapping=raw_mapping,
                        stage_dir=stage_dir,
                        async_transfer=async_transfer,
                    )
            if dist.is_initialized():
                dist.barrier(group=cast(Any, get_gloo_group()))
            return
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
            dist.barrier(group=cast(Any, get_gloo_group()))

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._sync_global_doc2k_votes()
        return super().state_dict(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        del kwargs
        if not self.gr_config.compact_train_ipc:
            self.decoder.gradient_checkpointing = True

    def _encode_prompts(
        self,
        sample_tokens: list[torch.Tensor],
        response_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, list[int], bool]:
        prompt_lengths = [
            tokens.shape[0] - response_length
            for tokens, response_length in zip(sample_tokens, response_lengths, strict=True)
        ]
        n = len(sample_tokens)
        g_size = DEFAULT_TRAIN_GROUP_SIZE
        is_uniform_grouped = (
            n >= g_size
            and n % g_size == 0
            and all(
                prompt_lengths[start] == prompt_lengths[i]
                for start in range(0, n, g_size)
                for i in range(start + 1, start + g_size)
            )
        )
        is_compact_grouped = (
            not is_uniform_grouped
            and n >= g_size
            and n % g_size == 0
            and all(
                prompt_lengths[i] == 1
                for start in range(0, n, g_size)
                for i in range(start + 1, start + g_size)
            )
        )
        group_inverse: torch.Tensor | None = None
        if is_uniform_grouped or is_compact_grouped:
            group_prompts = [
                sample_tokens[start][: prompt_lengths[start]].flip(0)
                for start in range(0, n, g_size)
            ]
            padded_groups = pad_sequence(
                group_prompts,
                batch_first=True,
                padding_value=self.pad_token_id,
            ).flip(1)
            unique_prompts, group_inv_t = torch.unique(padded_groups, dim=0, return_inverse=True)
            group_inverse = group_inv_t
            inverse_indices = group_inv_t.repeat_interleave(g_size)
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
            unique_prompts, inverse_indices = torch.unique(padded_prompts, dim=0, return_inverse=True)
        unique_mask = unique_prompts.ne(self.pad_token_id).long()
        outputs = self._encoder_backbone()(
            input_ids=unique_prompts,
            attention_mask=unique_mask,
            return_dict=True,
            use_cache=False,
        )
        return outputs.last_hidden_state, unique_mask, inverse_indices, group_inverse, prompt_lengths, is_compact_grouped

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

    def _extract_response_matrix(
        self,
        input_ids: torch.Tensor,
        sample_tokens: list[torch.Tensor],
        output_offsets: list[int],
        prompt_lengths: list[int],
        total_lengths: list[int],
        *,
        is_compact_grouped: bool,
    ) -> torch.Tensor:
        input_batch_size = input_ids.shape[0]
        sample_count = len(sample_tokens)
        g_size = DEFAULT_TRAIN_GROUP_SIZE
        if (
            input_batch_size == 1
            and is_compact_grouped
            and all(
                total_lengths[i] == self.code_num + 1
                for start in range(0, sample_count, g_size)
                for i in range(start + 1, start + g_size)
            )
        ):
            compact_len = self.code_num + 1
            chunks: list[torch.Tensor] = []
            for start in range(0, sample_count, g_size):
                off0 = output_offsets[start]
                plen0 = prompt_lengths[start]
                chunks.append(input_ids[0, off0 + plen0 : off0 + plen0 + self.code_num].unsqueeze(0))
                off1 = output_offsets[start + 1]
                chunks.append(
                    input_ids[0, off1 : off1 + (g_size - 1) * compact_len].view(g_size - 1, compact_len)[:, 1:]
                )
            return torch.cat(chunks, dim=0)
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
            return torch.cat(
                [
                    input_ids[
                        0,
                        output_offsets[start] : output_offsets[start] + g_size * total_lengths[start],
                    ].view(g_size, total_lengths[start])[:, prompt_lengths[start] : prompt_lengths[start] + self.code_num]
                    for start in range(0, sample_count, g_size)
                ],
                dim=0,
            )
        return torch.stack(
            [
                tokens[prompt_length : prompt_length + self.code_num]
                for tokens, prompt_length in zip(sample_tokens, prompt_lengths, strict=True)
            ]
        )

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

        (
            unique_hidden_states,
            unique_mask,
            inverse_indices,
            group_inverse,
            prompt_lengths,
            is_compact_grouped,
        ) = self._encode_prompts(sample_tokens, response_lengths)
        unique_projected = self.docEnc2decoder(unique_hidden_states.to(self.docEnc2decoder.weight.dtype))
        if self.gr_config.grouped_cross_attention and group_inverse is not None:
            projected_hidden_states = unique_projected.index_select(0, group_inverse)
            attention_mask = unique_mask.index_select(0, group_inverse)
        else:
            projected_hidden_states = unique_projected.index_select(0, inverse_indices)
            attention_mask = unique_mask.index_select(0, inverse_indices)

        input_batch_size, sequence_length = input_ids.shape
        sample_count = len(sample_tokens)
        vocab_size = self.code_num * self.code_dim + 2
        logits = projected_hidden_states.new_zeros((input_batch_size, sequence_length, vocab_size))

        responses = self._extract_response_matrix(
            input_ids,
            sample_tokens,
            output_offsets,
            prompt_lengths,
            total_lengths,
            is_compact_grouped=is_compact_grouped,
        )
        decoder_input_ids = torch.cat(
            (torch.zeros((sample_count, 1), dtype=torch.long, device=input_ids.device), responses[:, :-1]),
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
        allowed_mask = self._build_allowed_mask(responses, vocab_size)
        masked_response_logits = response_logits.masked_fill(~allowed_mask, _INVALID_DOCID_LOGIT)
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

"""SGLang policy model for SLMv3 query-to-docid rollout."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from examples.experimental.generative_retriever.constrained_decode import (
    build_generation_model,
    generate_beams,
    load_constraint,
)

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_classification import Qwen3ForPooledOutput


class GRQwen3ForDocIDGeneration(Qwen3ForPooledOutput):
    """Frozen Qwen3 encoder with an updateable SLMv3 Stage5 policy head."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        raw_gr_config = getattr(config, "gr_config", None)
        if not isinstance(raw_gr_config, dict):
            raise ValueError("GRQwen3ForDocIDGeneration requires config.gr_config")

        self.gr_config = T5Config.from_dict(raw_gr_config)
        self.docid_size = int(raw_gr_config["docid_size"])
        self.docid_dim = int(raw_gr_config["docid_dim"])
        self.num_beams = int(raw_gr_config["num_beams"])
        self.num_return_sequences = int(raw_gr_config["num_return_sequences"])
        if not 1 <= self.num_return_sequences <= self.num_beams:
            raise ValueError("Require 1 <= num_return_sequences <= num_beams")
        self.constraint = load_constraint(
            raw_gr_config["slm_root"], raw_gr_config["id_mapping"], self.docid_size, self.docid_dim,
        )

        decoder_config = self.gr_config
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = True
        decoder_config.use_cache = False
        decoder_config.num_layers = decoder_config.num_decoder_layers
        decoder_vocab_size = self.docid_size * self.docid_dim + 2
        decoder_config.vocab_size = decoder_vocab_size

        self.docEnc2decoder = nn.Linear(
            config.hidden_size,
            decoder_config.d_model,
            bias=False,
        )
        self.docid_embedding = nn.Embedding(
            decoder_vocab_size,
            decoder_config.d_model,
            padding_idx=decoder_config.pad_token_id,
        )
        self.decoder = T5Stack(decoder_config)
        self.decoder.embed_tokens = self.docid_embedding
        self.lm_head = nn.Linear(
            decoder_config.d_model,
            decoder_vocab_size,
            bias=False,
        )
        generation_model = build_generation_model(
            slm_root=raw_gr_config["slm_root"], config=decoder_config,
            projection=self.docEnc2decoder, decoder=self.decoder, lm_head=self.lm_head,
        )
        object.__setattr__(self, "_generation_model", generation_model)

    @staticmethod
    def _split_hidden_states(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix_lens = forward_batch.extend_prefix_lens_cpu or [0] * forward_batch.batch_size
        if any(prefix_lens):
            raise RuntimeError("GR policy requires --disable-radix-cache so every query has full hidden states")
        lengths = forward_batch.extend_seq_lens_cpu
        if not lengths or sum(lengths) != hidden_states.shape[0]:
            raise RuntimeError(
                f"invalid GR prefill lengths {lengths!r} for {hidden_states.shape[0]} hidden states"
            )

        sequences = hidden_states.split(lengths)
        max_length = max(lengths)
        padded = hidden_states.new_zeros((len(sequences), max_length, hidden_states.shape[-1]))
        attention_mask = torch.zeros(
            (len(sequences), max_length),
            dtype=torch.long,
            device=hidden_states.device,
        )
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.shape[0]] = sequence
            attention_mask[index, : sequence.shape[0]] = 1
        return padded, attention_mask

    def _generate_docids(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> EmbeddingPoolerOutput:
        token_ids, token_logprobs = generate_beams(
            self._generation_model, hidden_states, attention_mask, self.constraint,
            code_num=self.docid_size, num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
        )
        packed = torch.cat((token_ids.float(), token_logprobs.float()), dim=-1).flatten(1)
        return EmbeddingPoolerOutput(embeddings=packed)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        if not get_embedding:
            raise ValueError("GRQwen3ForDocIDGeneration only supports embedding-mode rollout")
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        hidden_states, attention_mask = self._split_hidden_states(hidden_states, forward_batch)
        return self._generate_docids(hidden_states, attention_mask)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        for original_name, loaded_weight in weights:
            if original_name.startswith("encoder_list.0.model.encoder."):
                qwen_name = original_name.removeprefix("encoder_list.0.model.encoder.")
                super().load_weights([(qwen_name, loaded_weight)])
                continue

            name = original_name
            if name in {"docid_embedding.weight", "decoder.embed_tokens.weight"}:
                target_name = "docid_embedding.weight"
            elif name.startswith(("docEnc2decoder.", "decoder.", "lm_head.")):
                target_name = name
            else:
                continue

            parameter = params.get(target_name)
            if parameter is None:
                raise KeyError(f"GR rollout parameter not found: {target_name}")
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, loaded_weight)


EntryClass = GRQwen3ForDocIDGeneration

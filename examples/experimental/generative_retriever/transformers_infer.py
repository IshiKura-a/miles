"""Load an SLMv3 generative retriever and run query-to-docid inference.

The checkpoint, Qwen backbone, and SLMv3 source tree are explicit inputs. This
module intentionally reuses the original SLMv3 model classes so checkpoint
loading and generation stay aligned with training.

Example:
    python -m examples.experimental.generative_retriever.transformers_infer \
        --checkpoint /path/to/checkpoint \
        --qwen-model /path/to/Qwen3-4B \
        --slm-root /path/to/SLMv3 \
        --transformers-path /path/to/transformers-4.57.3 \
        --query "cheap accommodation oporto with global travelers"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model_loading import (
    build_run_arguments,
    checkpoint_tensor_shape,
    import_slmv3,
    load_checkpoint,
    require_directory,
)


@dataclass(frozen=True)
class GenerationResult:
    query: str
    prompt: str
    decoder_token_ids: list[int]
    positional_docid: list[int]
    rq_codes: list[int]
    token_logprobs: list[float]


def _prepare_transformers(checkpoint: Path, transformers_path: str | Path | None) -> None:
    if transformers_path is not None:
        if "transformers" in sys.modules:
            raise RuntimeError("transformers_path must be configured before importing transformers")
        source_path = require_directory(transformers_path, "Transformers source")
        sys.path.insert(0, str(source_path))

    import transformers

    config_path = checkpoint / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        expected_version = json.load(config_file).get("transformers_version")
    if expected_version and transformers.__version__ != expected_version:
        raise RuntimeError(
            f"checkpoint requires transformers=={expected_version}, "
            f"but loaded {transformers.__version__}; pass transformers_path with the matching source"
        )


class SLMv3Retriever:
    """Original Transformers implementation of the SLMv3 query-to-docid path."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        qwen_model: str | Path,
        slm_root: str | Path,
        transformers_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        self.checkpoint = require_directory(checkpoint, "checkpoint")
        self.qwen_model = require_directory(qwen_model, "Qwen model")
        self.slm_root = require_directory(slm_root, "SLMv3 source")
        _prepare_transformers(self.checkpoint, transformers_path)

        import torch
        from transformers import T5Config, T5ForConditionalGeneration

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r}, but no CUDA/ROCm device is available")
        self.device = torch.device(device)

        slmv3 = import_slmv3(self.slm_root)
        run_args = build_run_arguments(slmv3["RunArguments"], self.checkpoint, self.qwen_model)
        t5_config = T5Config.from_pretrained(self.checkpoint)
        query_embedding_shape = checkpoint_tensor_shape(
            self.checkpoint,
            "query_encoder.embed_tokens.weight",
        )
        t5_config.vocab_size = query_embedding_shape[0]
        t5_model = T5ForConditionalGeneration(t5_config)
        qwen_encoder = slmv3["Encoder_QWen1216t"](run_args)
        self.model = slmv3["T5MultiWithGivenQueryDocEncoderDSI"](
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
        load_checkpoint(self.model, self.checkpoint)
        self.model.eval().to(device=self.device, dtype=torch.bfloat16)
        self.encoder = self.model.encoder_list[0]
        self.processor = self.encoder.processor
        self.tokenizer = self.encoder.tokenizer
        self.code_dim = run_args.code_dim
        self.code_num = run_args.code_num

    def _render_prompt(self, query: str) -> str:
        truncated_query = self.tokenizer.decode(
            self.tokenizer.encode(
                query,
                max_length=16,
                truncation=True,
                add_special_tokens=False,
            )
        )
        query_instruction = f'Given the user Query: "{truncated_query}", summarize it in one token:'
        return self.processor.apply_chat_template(
            [{"role": "user", "content": query_instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _allowed_tokens(self, batch_index: int, prefix: Any) -> list[int]:
        del batch_index
        position = prefix.shape[0] - 1
        start = 2 + position * self.code_dim
        return list(range(start, start + self.code_dim))

    def generate(
        self,
        queries: list[str],
        *,
        num_beams: int = 1,
        num_return_sequences: int = 1,
    ) -> list[GenerationResult]:
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

        if not queries or any(not isinstance(query, str) or not query.strip() for query in queries):
            raise ValueError("queries must contain at least one non-empty string")
        if num_beams <= 0 or num_return_sequences <= 0 or num_return_sequences > num_beams:
            raise ValueError("require num_beams > 0 and 0 < num_return_sequences <= num_beams")

        prompts = [self._render_prompt(query) for query in queries]
        batch = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in batch.items()}

        with torch.inference_mode():
            _, hidden_states, attention_mask = self.encoder.infer_embedding(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                },
                return_hidden_states=True,
            )
            encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=hidden_states,
                attentions=attention_mask,
            )
            generated = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                max_length=self.code_num + 1,
                min_length=self.code_num + 1,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                prefix_allowed_tokens_fn=self._allowed_tokens,
                early_stopping=True,
                forced_bos_token_id=None,
                forced_eos_token_id=None,
                decoder_start_token_id=0,
                return_dict_in_generate=True,
                output_scores=True,
            )
            transition_scores = self.model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                getattr(generated, "beam_indices", None),
                normalize_logits=True,
            )

        results = []
        for output_index, sequence in enumerate(generated.sequences):
            query_index = output_index // num_return_sequences
            decoder_token_ids = sequence[1:].tolist()
            positional_docid = [token_id - 2 for token_id in decoder_token_ids]
            rq_codes = [
                token_id - 2 - position * self.code_dim
                for position, token_id in enumerate(decoder_token_ids)
            ]
            results.append(
                GenerationResult(
                    query=queries[query_index],
                    prompt=prompts[query_index],
                    decoder_token_ids=decoder_token_ids,
                    positional_docid=positional_docid,
                    rq_codes=rq_codes,
                    token_logprobs=transition_scores[output_index].tolist(),
                )
            )
        return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--slm-root", required=True)
    parser.add_argument("--transformers-path")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    retriever = SLMv3Retriever(
        checkpoint=args.checkpoint,
        qwen_model=args.qwen_model,
        slm_root=args.slm_root,
        transformers_path=args.transformers_path,
        device=args.device,
    )
    results = retriever.generate(
        args.query,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
    )
    print(json.dumps([result.__dict__ for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

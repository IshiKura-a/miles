"""Standalone zero-repo-dependency loader and inference example for exported SLMv3 checkpoints.

Requires only `torch` and `transformers`:

    python standalone_slmv3_infer.py --ckpt-dir /path/to/exported/checkpoint-234
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5Config, T5ForConditionalGeneration

VOCAB_SIZE = 1026
DECODER_STEPS = 4
CODES_PER_STEP = 256
CODE_TOKEN_OFFSET = 2
QUERY_INNER_MAX_TOKENS = 16
PROMPT_MAX_TOKENS = 64
COPY_BUFFER_BYTES = 16 * 1024 * 1024

QWEN_WEIGHT_PREFIX = "encoder_list.0.model.encoder.model."
HEAD_WEIGHT_PREFIXES = ("docEnc2decoder.", "decoder.", "lm_head.")
DEFAULT_EXAMPLE_QUERIES: tuple[str, ...] = (
    "women running shoes",
    "cheap flights to tokyo",
)
QWEN_CONFIG_ASSETS: tuple[str, ...] = (
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


class GenerativeRetriever(nn.Module):
    def __init__(self, qwen_cfg: Any, t5_cfg: T5Config) -> None:
        super().__init__()
        self.qwen = AutoModel.from_config(qwen_cfg)
        self.docEnc2decoder = nn.Linear(qwen_cfg.hidden_size, t5_cfg.d_model, bias=False)
        self.decoder = T5ForConditionalGeneration(t5_cfg).decoder
        self.lm_head = nn.Linear(t5_cfg.d_model, t5_cfg.vocab_size, bias=False)


def stream_copy_file(src: Path, dst: Path) -> None:
    with src.open("rb") as src_f, dst.open("wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f, length=COPY_BUFFER_BYTES)


def load_qwen_config_and_tokenizer(ckpt_dir: Path) -> tuple[Any, Any]:
    qwen_dir = ckpt_dir / "qwen_config"
    with tempfile.TemporaryDirectory(prefix="slmv3_qwen_cfg_") as tmp_str:
        tmp_dir = Path(tmp_str)
        for name in QWEN_CONFIG_ASSETS:
            stream_copy_file(qwen_dir / name, tmp_dir / name)
        qwen_cfg = AutoConfig.from_pretrained(tmp_dir, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(
            tmp_dir,
            padding_side="left",
            truncation_side="right",
            local_files_only=True,
        )
    return qwen_cfg, tokenizer


def load_t5_config(ckpt_dir: Path) -> T5Config:
    with tempfile.TemporaryDirectory(prefix="slmv3_t5_cfg_") as tmp_str:
        local_cfg = Path(tmp_str) / "config.json"
        stream_copy_file(ckpt_dir / "config.json", local_cfg)
        t5_cfg = T5Config.from_json_file(local_cfg)
    t5_cfg.vocab_size = VOCAB_SIZE
    return t5_cfg


def load_retriever(
    ckpt_dir: str | Path,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[GenerativeRetriever, Any]:
    resolved_dir = Path(ckpt_dir).expanduser().resolve()
    qwen_cfg, tokenizer = load_qwen_config_and_tokenizer(resolved_dir)
    t5_cfg = load_t5_config(resolved_dir)

    with (resolved_dir / "pytorch_model.bin.index.json").open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shards:
        shard = torch.load(resolved_dir / shard_name, map_location="cpu", weights_only=True)
        for key, tensor in shard.items():
            if key.startswith(QWEN_WEIGHT_PREFIX):
                state_dict["qwen." + key[len(QWEN_WEIGHT_PREFIX) :]] = tensor
            elif key.startswith(HEAD_WEIGHT_PREFIXES):
                state_dict[key] = tensor

    model = GenerativeRetriever(qwen_cfg, t5_cfg).to(dtype=dtype)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), tokenizer


def format_query_prompts(tokenizer: Any, queries: list[str]) -> list[str]:
    prompts: list[str] = []
    for query in queries:
        truncated_query = tokenizer.decode(
            tokenizer.encode(
                query,
                max_length=QUERY_INNER_MAX_TOKENS,
                truncation=True,
                add_special_tokens=False,
            )
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": f'Given the user Query: "{truncated_query}", summarize it in one token:'}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


@torch.inference_mode()
def generate_docids(
    model: GenerativeRetriever,
    tokenizer: Any,
    queries: list[str],
    *,
    beam_size: int = 4,
) -> list[list[list[int]]]:
    device = next(model.parameters()).device
    prompts = format_query_prompts(tokenizer, queries)
    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=PROMPT_MAX_TOKENS,
        return_tensors="pt",
    ).to(device)
    enc_hs = model.docEnc2decoder(
        model.qwen(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    )

    bsz = len(queries)
    seqs = torch.zeros((bsz, 1, 1), dtype=torch.long, device=device)
    cum_lp = torch.zeros((bsz, 1), dtype=torch.float32, device=device)

    for step in range(DECODER_STEPS):
        cur_b = seqs.shape[1]
        dec_out = model.decoder(
            input_ids=seqs.reshape(bsz * cur_b, step + 1),
            encoder_hidden_states=enc_hs.repeat_interleave(cur_b, dim=0),
            encoder_attention_mask=batch["attention_mask"].repeat_interleave(cur_b, dim=0),
            use_cache=False,
        )
        logits = model.lm_head(dec_out.last_hidden_state[:, -1, :]).float()
        lo = CODE_TOKEN_OFFSET + CODES_PER_STEP * step
        hi = CODE_TOKEN_OFFSET + CODES_PER_STEP * (step + 1)

        # Replace `mask[:, lo:hi] = True` with prefix-tree child masks when constrained to an index.
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, lo:hi] = True
        step_lp = F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1).reshape(bsz, cur_b, -1)

        cand = (cum_lp.unsqueeze(-1) + step_lp).reshape(bsz, -1)
        k = min(beam_size, CODES_PER_STEP ** (step + 1))
        cum_lp, top_idx = torch.topk(cand, k=k, dim=-1)
        beam_idx = top_idx // logits.shape[-1]
        tok_idx = top_idx % logits.shape[-1]
        seqs = torch.cat(
            [seqs.gather(1, beam_idx.unsqueeze(-1).expand(bsz, k, step + 1)), tok_idx.unsqueeze(-1)],
            dim=-1,
        )

    return (seqs[:, :, 1:] - CODE_TOKEN_OFFSET).cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True, help="Exported SLMv3 checkpoint directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--beam-size", type=int, default=4, help="Beam size per query")
    parser.add_argument("--query", action="append", default=None, help="Query text to run (repeatable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = args.query if args.query is not None else list(DEFAULT_EXAMPLE_QUERIES)
    model, tokenizer = load_retriever(args.ckpt_dir, device=args.device)
    docids = generate_docids(model, tokenizer, queries, beam_size=args.beam_size)
    for query, beams in zip(queries, docids, strict=True):
        print(f"Query: {query!r}\n  top_docids={beams}")


if __name__ == "__main__":
    main()

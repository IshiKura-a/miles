"""Shared SLMv3 model-construction and checkpoint-loading utilities."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any


def require_directory(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    return resolved


def import_slmv3(slm_root: Path) -> dict[str, Any]:
    import transformers

    # Transformers 5 removed the MT5 aliases used by the original SLMv3 imports.
    if not hasattr(transformers, "MT5Tokenizer"):
        transformers.MT5Tokenizer = transformers.T5Tokenizer
    if not hasattr(transformers, "MT5TokenizerFast"):
        transformers.MT5TokenizerFast = transformers.T5TokenizerFast

    root_string = str(slm_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    model_module = importlib.import_module("model")
    utils_module = importlib.import_module("utils")
    encoder_module = importlib.import_module("CustomizedEncoder")
    return {
        "RunArguments": utils_module.RunArguments,
        "T5MultiWithGivenQueryDocEncoderDSI": model_module.T5MultiWithGivenQueryDocEncoderDSI,
        "Encoder_QWen1216t": encoder_module.Encoder_QWen1216t,
    }


def build_run_arguments(run_arguments_class: type, checkpoint: Path, qwen_model: Path) -> Any:
    return run_arguments_class(
        model_name=str(checkpoint),
        encoder_name="",
        tokenizer_name="",
        encoder_scratch=0,
        decoder_scratch=1,
        code_dim=256,
        code_num=4,
        lm_nobias=1,
        multi_tower=1,
        infonce=10,
        infonce_temp=0.05,
        cont_position="direct",
        ib=0,
        ib_strategy="gaussian",
        domain_count=1,
        training_job="train_llm_ShareAdaptor",
        customized_doc_encoder="Encoder_QWen1216t",
        customized_doc_encoder_ckpt=str(qwen_model),
        customized_doc_encoder_for_query=1,
        quant_rqvae=1,
        quant_rqvae_weight=1.0,
        quant_rqvae_type=2,
        quant_kmeans_init=0,
        train_stage_1=0,
        train_stage_2=0,
        train_stage_3=0,
        train_stage_4=0,
        train_stage_5=999999999,
    )


def checkpoint_shards(checkpoint: Path) -> list[Path]:
    index_path = checkpoint / "pytorch_model.bin.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")
    shards = sorted({checkpoint / filename for filename in weight_map.values()})
    missing = [shard for shard in shards if not shard.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint shards are missing: {missing}")
    return shards


def checkpoint_tensor_shape(checkpoint: Path, tensor_name: str) -> tuple[int, ...]:
    import torch

    index_path = checkpoint / "pytorch_model.bin.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index is missing: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        shard_name = json.load(index_file).get("weight_map", {}).get(tensor_name)
    if not shard_name:
        raise KeyError(f"checkpoint index does not contain tensor {tensor_name!r}")
    state_dict = torch.load(
        checkpoint / shard_name,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    tensor = state_dict.get(tensor_name)
    if tensor is None:
        raise KeyError(f"checkpoint shard {shard_name!r} does not contain tensor {tensor_name!r}")
    return tuple(tensor.shape)


def load_checkpoint(model: Any, checkpoint: Path) -> None:
    import torch

    loaded_keys: set[str] = set()
    for shard_path in checkpoint_shards(checkpoint):
        state_dict = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(f"checkpoint shard is not a state dict: {shard_path}")
        model.load_state_dict(state_dict, strict=False)
        loaded_keys.update(state_dict)

    required_prefixes = (
        "encoder_list.0.model.encoder.model.",
        "docEnc2decoder.",
        "decoder.",
        "docid_embedding.",
        "lm_head.",
    )
    missing_prefixes = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in loaded_keys)
    ]
    if missing_prefixes:
        raise ValueError(f"checkpoint is missing required model components: {missing_prefixes}")

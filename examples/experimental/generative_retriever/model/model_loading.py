"""Shared SLMv3 model-construction and checkpoint-loading utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import load

_POLICY_HEAD_PREFIXES = ("docEnc2decoder.", "decoder.", "lm_head.")


def require_directory(path: str | Path, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    return resolved


def import_slmv3(slm_root: Path) -> dict[str, Any]:
    root_string = str(slm_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    model_module = importlib.import_module("model")

    # Model imports can replace Transformers' lazy module; patch the active module before utils.
    transformers_dynamic: Any = importlib.import_module("transformers")
    if not hasattr(transformers_dynamic, "MT5Tokenizer"):
        transformers_dynamic.MT5Tokenizer = transformers_dynamic.T5Tokenizer
    if not hasattr(transformers_dynamic, "MT5TokenizerFast"):
        transformers_dynamic.MT5TokenizerFast = transformers_dynamic.T5TokenizerFast

    utils_module = importlib.import_module("utils")
    encoder_module = importlib.import_module("CustomizedEncoder")
    return {
        "RunArguments": utils_module.RunArguments,
        "T5MultiWithGivenQueryDocEncoderDSI": model_module.T5MultiWithGivenQueryDocEncoderDSI,
        "Encoder_QWen1216t": encoder_module.Encoder_QWen1216t,
        "DataSupport_QWen1216t": encoder_module.DataSupport_QWen1216t,
        "get_restrict_decode_vocab": utils_module.get_restrict_decode_vocab,
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


def load_policy_head_checkpoint(model: Any, checkpoint: str | Path) -> int:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        state_dict = payload["model_state"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Policy checkpoint has an invalid state-dict structure: {path}") from error
    selected = {
        name: tensor
        for name, tensor in state_dict.items()
        if name.startswith(_POLICY_HEAD_PREFIXES)
    }
    expected = {
        name for name in model.state_dict() if name.startswith(_POLICY_HEAD_PREFIXES)
    }
    missing = expected - selected.keys()
    unexpected = selected.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"Policy checkpoint head keys mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(selected, strict=False)
    return len(selected)


def export_policy_head_checkpoint(
    dcp_model_dir: str | Path,
    output_path: str | Path,
) -> int:
    ckpt_dir = require_directory(dcp_model_dir, "DCP model checkpoint")
    metadata_file = ckpt_dir / ".metadata"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"DCP checkpoint metadata is missing: {metadata_file}")
    out_pt = Path(output_path).expanduser().resolve()
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    dcp_prefix = "model_state.model."
    prefixes = tuple(dcp_prefix + head for head in _POLICY_HEAD_PREFIXES)
    with tempfile.TemporaryDirectory(prefix="gr_dcp_head_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        shutil.copyfile(metadata_file, tmp_dir / ".metadata")
        reader = FileSystemReader(tmp_dir)
        meta = reader.read_metadata()
        max_end: dict[str, int] = {}
        for key, item in meta.storage_data.items():
            if key.fqn.startswith(prefixes):
                max_end[item.relative_path] = max(
                    max_end.get(item.relative_path, 0),
                    item.offset + item.length,
                )

        def _copy_shard_prefix(rel_and_end: tuple[str, int]) -> None:
            rel_path, end_offset = rel_and_end
            with (ckpt_dir / rel_path).open("rb") as src, (tmp_dir / rel_path).open("wb") as dst:
                dst.write(src.read(end_offset))

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(max_end)))) as executor:
            list(executor.map(_copy_shard_prefix, sorted(max_end.items())))

        entries = {
            name[len(dcp_prefix):]: spec
            for name, spec in meta.state_dict_metadata.items()
            if name.startswith(prefixes) and isinstance(spec, TensorStorageMetadata)
        }
        if not entries:
            raise ValueError(f"No policy head tensors found in {ckpt_dir}")
        tensors = {
            name: torch.empty(tuple(spec.size), dtype=spec.properties.dtype)
            for name, spec in entries.items()
        }
        payload = {"model_state": {"model": tensors}}
        load(payload, storage_reader=reader)
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
            raise RuntimeError(f"Exported policy head contains non-finite weights: {ckpt_dir}")
        tmp_pt = tmp_dir / "policy_head.pt"
        torch.save(payload, tmp_pt)
        shutil.copy2(tmp_pt, out_pt)
    return len(tensors)


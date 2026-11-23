"""Shared SLMv3 model-construction and checkpoint-loading utilities."""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path, stream_copy_file

from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from filelock import FileLock
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.checkpoint.state_dict_loader import load

_POLICY_ENCODER_PREFIX = "encoder_list.0.model.encoder.model."
_POLICY_HEAD_PREFIXES = ("docEnc2decoder.", "decoder.", "lm_head.")
_POLICY_EXPORT_PREFIXES = (_POLICY_ENCODER_PREFIX, *_POLICY_HEAD_PREFIXES)
_SLMV3_TIED_WEIGHT_TARGETS: tuple[tuple[str, str], ...] = (
    ("encoder_list.0.model.encoder.lm_head.weight", "encoder_list.0.model.encoder.model.embed_tokens.weight"),
    ("docid_embedding.weight", "decoder.embed_tokens.weight"),
    ("glue.embeds", "decoder.embed_tokens.weight"),
)
_SLMV3_EXPORT_ASSETS = (
    "config.json",
    "customized_preprocess_fill_prompt.py",
    "generation_config.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
)
_QWEN_EXPORT_SUBDIR = "qwen_config"
_QWEN_EXPORT_ASSETS = (
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
_WEIGHT_SUFFIXES = (".bin", ".safetensors", ".pt", ".pth", ".onnx", ".tar")
_STAGE_ASSETS = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "config.json",
    "customized_preprocess_fill_prompt.py",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def require_directory(path: str | Path, name: str) -> Path:
    resolved = absolute_path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} directory does not exist: {resolved}")
    return resolved


def stage_local_model_dir(path: str | Path, name: str) -> Path:
    resolved = require_directory(path, name)
    local_root = Path(tempfile.gettempdir())
    if resolved.is_relative_to(local_root):
        return resolved
    digest = hashlib.sha256(os.fspath(resolved).encode("utf-8")).hexdigest()[:12]
    target = local_root / f"gr_model_{resolved.name}_{digest}"
    sentinel = target / ".staged_ok"
    index_names = ("pytorch_model.bin.index.json", "model.safetensors.index.json")
    if not sentinel.is_file():
        with FileLock(f"{target}.lock"):
            if not sentinel.is_file():
                with tempfile.TemporaryDirectory(prefix=f".gr_model_{resolved.name}_{digest}.", dir=local_root) as tmp:
                    staging = Path(tmp)
                    for asset_name in _STAGE_ASSETS:
                        src = resolved / asset_name
                        if src.is_file():
                            shutil.copyfile(src, staging / asset_name)
                    indexes = [staging / index_name for index_name in index_names if (staging / index_name).is_file()]
                    if not indexes:
                        raise FileNotFoundError(f"{name} has no model weight index: {resolved}")
                    for index in indexes:
                        with index.open(encoding="utf-8") as handle:
                            weight_map = json.load(handle)["weight_map"]
                        if not weight_map:
                            raise ValueError(f"Model weight index is empty: {index}")
                        for shard_name in sorted(set(weight_map.values())):
                            src = resolved / shard_name
                            if not src.is_file():
                                raise FileNotFoundError(f"Model shard is missing: {src}")
                            dst = staging / shard_name
                            if not dst.is_symlink():
                                dst.symlink_to(src)
                    (staging / ".staged_ok").write_text(os.fspath(resolved) + "\n", encoding="utf-8")
                    os.replace(staging, target)
    if sentinel.read_text(encoding="utf-8").strip() != os.fspath(resolved) or not any(
        (target / index_name).is_file() for index_name in index_names
    ):
        raise ValueError(f"Invalid staged model directory: {target}")
    return target


def import_slmv3(slm_root: Path) -> dict[str, Any]:
    root_string = str(slm_root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    model_module = importlib.import_module("model")

    # Transformers 5 maps mT5 to T5Tokenizer; the SLM source imports the old names.
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
    shards = checkpoint_shards(checkpoint)

    def _read_shard(shard_path: Path) -> dict[str, Any]:
        sd = torch.load(shard_path, map_location="cpu", weights_only=True)
        if not isinstance(sd, dict):
            raise TypeError(f"checkpoint shard is not a state dict: {shard_path}")
        return sd

    loaded_keys: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(shards)))) as pool:
        for state_dict in pool.map(_read_shard, shards):
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
    path = absolute_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Policy checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    try:
        state_dict = payload["model_state"]["model"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Policy checkpoint has an invalid state-dict structure: {path}") from error
    has_encoder = any(name.startswith(_POLICY_ENCODER_PREFIX) for name in state_dict)
    active_prefixes = _POLICY_EXPORT_PREFIXES if has_encoder else _POLICY_HEAD_PREFIXES
    selected = {
        name: tensor
        for name, tensor in state_dict.items()
        if name.startswith(active_prefixes)
    }
    expected = {
        name for name in model.state_dict() if name.startswith(active_prefixes)
    }
    missing = expected - selected.keys()
    unexpected = selected.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"Policy checkpoint keys mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    model.load_state_dict(selected, strict=False)
    return len(selected)


def _load_dcp_policy_tensors(dcp_model_dir: str | Path) -> dict[str, torch.Tensor]:
    ckpt_dir = require_directory(dcp_model_dir, "DCP model checkpoint")
    metadata_file = ckpt_dir / ".metadata"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"DCP checkpoint metadata is missing: {metadata_file}")

    dcp_prefix = "model_state.model."
    prefixes = tuple(dcp_prefix + head for head in _POLICY_EXPORT_PREFIXES)
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
            name[len(dcp_prefix) :]: spec
            for name, spec in meta.state_dict_metadata.items()
            if name.startswith(prefixes) and isinstance(spec, TensorStorageMetadata)
        }
        if not entries:
            raise ValueError(f"No policy head tensors found in {ckpt_dir}")
        tensors = {
            name: torch.empty(tuple(spec.size), dtype=spec.properties.dtype)
            for name, spec in entries.items()
        }
        load({"model_state": {"model": tensors}}, storage_reader=reader)
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors.values()):
            raise RuntimeError(f"Exported policy tensors contain non-finite weights: {ckpt_dir}")
        return tensors


def export_policy_head_checkpoint(
    dcp_model_dir: str | Path,
    output_path: str | Path,
) -> int:
    out_pt = absolute_path(output_path)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    tensors = _load_dcp_policy_tensors(dcp_model_dir)
    payload = {"model_state": {"model": tensors}}
    with tempfile.TemporaryDirectory(prefix="gr_policy_pt_") as tmp_dir_str:
        tmp_pt = Path(tmp_dir_str) / "policy_head.pt"
        torch.save(payload, tmp_pt)
        stream_copy_file(tmp_pt, out_pt)
    return len(tensors)


def _build_slmv3_weight_updates(
    policy_tensors: dict[str, torch.Tensor],
    weight_map: dict[str, str],
) -> dict[str, torch.Tensor]:
    expected_policy_keys = {key for key in weight_map if key.startswith(_POLICY_EXPORT_PREFIXES)}
    missing = expected_policy_keys - policy_tensors.keys()
    unexpected = policy_tensors.keys() - expected_policy_keys
    if missing or unexpected:
        raise ValueError(
            f"DCP policy keys do not match base checkpoint: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    updates = dict(policy_tensors)
    for target_key, source_key in _SLMV3_TIED_WEIGHT_TARGETS:
        if target_key not in weight_map:
            raise KeyError(f"Base checkpoint index is missing tied target {target_key!r}")
        if source_key not in policy_tensors:
            raise KeyError(f"DCP policy checkpoint is missing tied source {source_key!r}")
        updates[target_key] = policy_tensors[source_key]
    return updates


def _merge_slmv3_shard(
    shard_path: Path,
    expected_keys: set[str],
    updates: dict[str, torch.Tensor],
    output_shard_path: Path,
) -> None:
    orig_shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    if not isinstance(orig_shard, dict):
        raise TypeError(f"Base checkpoint shard is not a state dict: {shard_path}")
    if set(orig_shard.keys()) != expected_keys:
        raise ValueError(f"Base checkpoint shard keys mismatch index for {shard_path.name}")

    merged: dict[str, torch.Tensor] = {}
    for key, orig_tensor in orig_shard.items():
        if key not in updates:
            merged[key] = orig_tensor
            continue
        new_tensor = updates[key]
        if tuple(new_tensor.shape) != tuple(orig_tensor.shape):
            raise ValueError(
                f"Shape mismatch for {key!r}: DCP {tuple(new_tensor.shape)} vs base {tuple(orig_tensor.shape)}"
            )
        if not orig_tensor.is_floating_point():
            raise TypeError(f"Expected floating-point parameter for updated key {key!r}, got {orig_tensor.dtype}")
        merged[key] = new_tensor.to(dtype=orig_tensor.dtype).contiguous()
    torch.save(merged, output_shard_path)


def copy_qwen_export_assets(qwen_model_dir: str | Path, output_dir: str | Path) -> int:
    raw_qwen = require_directory(qwen_model_dir, "Qwen model")
    staged_qwen = stage_local_model_dir(raw_qwen, "Qwen model")
    qwen_out = absolute_path(output_dir) / _QWEN_EXPORT_SUBDIR
    qwen_out.mkdir(parents=True, exist_ok=True)
    for asset_name in _QWEN_EXPORT_ASSETS:
        src_asset = staged_qwen / asset_name if (staged_qwen / asset_name).is_file() else raw_qwen / asset_name
        if not src_asset.is_file():
            raise FileNotFoundError(f"Qwen config/tokenizer asset is missing: {raw_qwen / asset_name}")
        stream_copy_file(src_asset, qwen_out / asset_name)
    return len(_QWEN_EXPORT_ASSETS)


def export_slmv3_checkpoint(
    dcp_model_dir: str | Path,
    *,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    qwen_model_dir: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    """Export an FSDP DCP checkpoint into a standalone SLMv3 checkpoint directory."""
    raw_base = require_directory(base_checkpoint, "base checkpoint")
    staged_base = stage_local_model_dir(raw_base, "base checkpoint")
    out_dir = absolute_path(output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out_dir}")
        shutil.rmtree(out_dir)

    for asset_name in _SLMV3_EXPORT_ASSETS:
        if not (raw_base / asset_name).is_file():
            raise FileNotFoundError(f"Base checkpoint asset is missing: {raw_base / asset_name}")

    index_data = json.loads((staged_base / "pytorch_model.bin.index.json").read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid weight_map in {raw_base / 'pytorch_model.bin.index.json'}")

    policy_tensors = _load_dcp_policy_tensors(dcp_model_dir)
    updates = _build_slmv3_weight_updates(policy_tensors, weight_map)

    keys_by_shard: dict[str, set[str]] = {}
    for tensor_key, shard_name in weight_map.items():
        if not (shard_name.startswith("pytorch_model") and shard_name.endswith(".bin")):
            raise ValueError(f"Unexpected SLMv3 shard filename {shard_name!r} for {tensor_key!r}")
        keys_by_shard.setdefault(shard_name, set()).add(tensor_key)

    with tempfile.TemporaryDirectory(prefix="gr_slmv3_export_") as tmp_dir_str:
        tmp_out = Path(tmp_dir_str)
        for asset_name in _SLMV3_EXPORT_ASSETS:
            src_asset = staged_base / asset_name if (staged_base / asset_name).is_file() else raw_base / asset_name
            shutil.copyfile(src_asset, tmp_out / asset_name)
        for shard_name, shard_keys in sorted(keys_by_shard.items()):
            _merge_slmv3_shard(staged_base / shard_name, shard_keys, updates, tmp_out / shard_name)

        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in sorted(p.name for p in tmp_out.iterdir()):
            stream_copy_file(tmp_out / filename, out_dir / filename)

    qwen_assets = copy_qwen_export_assets(qwen_model_dir, out_dir) if qwen_model_dir is not None else 0
    return {
        "total_tensors": len(weight_map),
        "updated_policy_tensors": len(policy_tensors),
        "synced_tied_tensors": len(_SLMV3_TIED_WEIGHT_TARGETS),
        "shards": len(keys_by_shard),
        "qwen_config_assets": qwen_assets,
    }


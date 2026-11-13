from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)


class ModelState(Stateful):
    """Wrapper for model state only."""

    def __init__(self, model):
        self.model = model

    def state_dict(self):
        model_state_dict, _ = get_state_dict(self.model, optimizers=[])
        if getattr(self.model, "_miles_fsdp_ignore_frozen_parameters", False):
            trainable_keys = {
                name for name, param in self.model.named_parameters() if param.requires_grad
            }
            model_state_dict = {
                key: value for key, value in model_state_dict.items() if key in trainable_keys
            }
        return {"model": model_state_dict}

    def load_state_dict(self, state_dict):
        strict = not getattr(self.model, "_miles_fsdp_ignore_frozen_parameters", False)
        set_state_dict(
            self.model,
            optimizers=[],
            model_state_dict=state_dict["model"],
            optim_state_dict=None,
            options=StateDictOptions(strict=strict),
        )


class OptimizerState(Stateful):
    """Wrapper for optimizer state only."""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        _, optimizer_state_dict = get_state_dict(self.model, optimizers=self.optimizer)
        return {"optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model, optimizers=self.optimizer, model_state_dict=None, optim_state_dict=state_dict["optim"]
        )


class LRSchedulerState(Stateful):
    """Wrapper for LR scheduler state only."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def state_dict(self):
        return {"lr_scheduler": self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse checkpoint metadata at {path}")
        return {}


def _write_checkpoint_metadata(path: Path, metadata: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    tmp_path.replace(path)


def load(actor: Any) -> dict[str, Any] | None:
    """Load checkpoint from disk.

    Loads model weights and optionally optimizer state from separate directories.
    This allows loading weights without optimizer or deleting optimizer before loading.
    """
    load_root = getattr(actor.args, "load", None)
    if load_root is None:
        return None

    root_path = Path(load_root).expanduser()
    if not root_path.exists():
        logger.info(f"[FSDP] Checkpoint directory {root_path} not found; skipping load.")
        return None

    target_step = getattr(actor.args, "ckpt_step", None)
    if target_step is None:
        tracker_file = root_path / "latest_checkpointed_iteration.txt"
        if not tracker_file.exists():
            logger.info(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None
        tracker_text = tracker_file.read_text().strip()
        target_step = int(tracker_text)

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        logger.info(f"[FSDP] Model checkpoint {model_dir} not found; skipping load.")
        return None

    # Load model weights (always)
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}

    try:
        dcp.load(state_dict=state_dict, checkpoint_id=str(model_dir))
        logger.info(f"[FSDP] Loaded model from {model_dir}")
    except Exception as e:
        logger.error(f"[FSDP] Failed to load model from {model_dir}: {e}")
        return None

    # Load optimizer state (optional)
    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        try:
            dcp.load(state_dict=optim_state_dict, checkpoint_id=str(optimizer_dir))
            logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load optimizer from {optimizer_dir}: {e}")
    elif load_optimizer:
        logger.info(f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    # Load LR scheduler state (optional)
    load_lr_scheduler = hasattr(actor, "lr_scheduler") and lr_scheduler_dir.exists()
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        try:
            dcp.load(state_dict=lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))
            logger.info(f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir}")
        except Exception as e:
            logger.warning(f"[FSDP] Failed to load LR scheduler from {lr_scheduler_dir}: {e}")
    elif hasattr(actor, "lr_scheduler"):
        logger.info(f"[FSDP] LR scheduler checkpoint not found at {lr_scheduler_dir}, skipping LR scheduler load.")

    rng_state = None
    rng_path = checkpoint_dir / "rng.pt"
    if rng_path.exists():
        rng_state = torch.load(rng_path, map_location="cpu")

    metadata = _read_checkpoint_metadata(checkpoint_dir / "meta.json")

    return {
        "rng": rng_state,
        "metadata": metadata,
        "iteration": target_step,
    }


def finalize_load(actor: Any, checkpoint_payload: dict[str, Any] | None) -> None:
    if checkpoint_payload is None:
        dist.barrier()
        return

    if checkpoint_payload.get("rng") is not None and not getattr(actor.args, "no_load_rng", False):
        rng_state = checkpoint_payload["rng"]
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    metadata = checkpoint_payload.get("metadata") or {}
    iteration = checkpoint_payload.get("iteration")
    if metadata:
        actor.global_step = int(metadata.get("global_step", actor.global_step))
        actor.micro_step = int(metadata.get("micro_step", actor.micro_step))
        next_rollout = metadata.get("next_rollout_id")
        if next_rollout is not None:
            actor.args.start_rollout_id = next_rollout
    elif iteration is not None:
        if getattr(actor.args, "start_rollout_id", None) is None:
            actor.args.start_rollout_id = iteration

    torch.cuda.synchronize()
    dist.barrier()


_ACTIVE_SYNC_THREADS: list[Any] = []


def flush_staged_saves(actor: Any | None = None) -> None:
    if actor is not None:
        t = getattr(actor, "_temp_stage_sync_thread", None)
        if t is not None and t.is_alive():
            t.join()
        if hasattr(actor, "_temp_stage_sync_thread"):
            actor._temp_stage_sync_thread = None
    while _ACTIVE_SYNC_THREADS:
        t = _ACTIVE_SYNC_THREADS.pop(0)
        if t is not None and t.is_alive():
            t.join()


import atexit as _atexit
_atexit.register(flush_staged_saves)


def save(actor: Any, iteration: int, force_sync: bool = False) -> None:
    """Save checkpoint to disk.

    When actor.args.temp_stage_save is True (default), all checkpoint artifacts
    (mapping, model, optimizer, lr_scheduler, rng.pt, meta.json) are written to fast
    local /tmp storage first and then synchronized to the target save directory
    in a background thread (or synchronously when force_sync=True).
    """
    import os
    import shutil
    import tempfile
    import threading
    from miles.utils.distributed_utils import get_gloo_group

    torch.cuda.synchronize()
    flush_staged_saves(actor)
    gloo_group = get_gloo_group()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    rank = dist.get_rank()
    temp_stage_save = bool(getattr(actor.args, "temp_stage_save", True))

    if temp_stage_save:
        stage_root = Path(tempfile.mkdtemp(prefix=f"miles_ckpt_s{step_id}_r{rank}_"))
        write_ckpt_dir = stage_root / f"iter_{step_id:07d}"
    else:
        stage_root = None
        write_ckpt_dir = checkpoint_dir

    model_dir = write_ckpt_dir / "model"
    optimizer_dir = write_ckpt_dir / "optimizer"
    lr_scheduler_dir = write_ckpt_dir / "lr_scheduler"

    if temp_stage_save or rank == 0:
        write_ckpt_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        optimizer_dir.mkdir(parents=True, exist_ok=True)
        lr_scheduler_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier(group=gloo_group)

    if hasattr(actor.model, "save_doc2k_mapping"):
        actor.model.save_doc2k_mapping(
            checkpoint_dir / "mapping",
            stage_dir=(write_ckpt_dir / "mapping") if temp_stage_save else None,
            async_transfer=not force_sync,
        )

    # Save model weights
    model_state = ModelState(actor.model)
    state_dict = {"model_state": model_state}
    dcp.save(state_dict, checkpoint_id=str(model_dir))

    # Save optimizer state (skip if --no-save-optim is set)
    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        optim_state_dict = {"optim_state": optimizer_state}
        dcp.save(optim_state_dict, checkpoint_id=str(optimizer_dir))

    # Save LR scheduler state (skip if --no-save-optim is set)
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        lr_scheduler_state_dict = {"lr_scheduler_state": lr_scheduler_state}
        dcp.save(lr_scheduler_state_dict, checkpoint_id=str(lr_scheduler_dir))

    if rank == 0:
        rng_state = {"torch": torch.get_rng_state()}
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(rng_state, write_ckpt_dir / "rng.pt")

        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": dist.get_world_size(),
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(write_ckpt_dir / "meta.json", metadata)

        if not temp_stage_save:
            tracker_file = base_dir / "latest_checkpointed_iteration.txt"
            tracker_file.write_text(str(step_id))
            logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")

    dist.barrier(group=gloo_group)

    if temp_stage_save and stage_root is not None:
        if rank == 0:
            for sub in ("model", "optimizer", "lr_scheduler", "mapping"):
                (checkpoint_dir / sub).mkdir(parents=True, exist_ok=True)
            logger.info(f"[FSDP] Staged checkpoint to local {write_ckpt_dir}; starting sync to {checkpoint_dir}")
        dist.barrier(group=gloo_group)

        def _sync_staged_to_target() -> None:
            t0 = time.perf_counter()
            try:
                for src_item in sorted(write_ckpt_dir.rglob("*")):
                    rel = src_item.relative_to(write_ckpt_dir)
                    dst_item = checkpoint_dir / rel
                    if src_item.is_symlink():
                        dst_item.parent.mkdir(parents=True, exist_ok=True)
                        if not dst_item.exists() and not dst_item.is_symlink():
                            dst_item.symlink_to(os.readlink(src_item))
                    elif src_item.is_dir():
                        dst_item.mkdir(parents=True, exist_ok=True)
                    elif src_item.is_file():
                        dst_item.parent.mkdir(parents=True, exist_ok=True)
                        dst_tmp = dst_item.with_name(f"{dst_item.name}.tmp.r{rank}")
                        shutil.copyfile(src_item, dst_tmp)
                        dst_tmp.replace(dst_item)
                if rank == 0:
                    tracker_tmp = base_dir / f"latest_checkpointed_iteration.txt.tmp.s{step_id}"
                    tracker_tmp.write_text(str(step_id))
                    tracker_tmp.replace(base_dir / "latest_checkpointed_iteration.txt")
                    logger.info(
                        "[FSDP] Synced staged checkpoint to %s in %.2fs",
                        checkpoint_dir,
                        time.perf_counter() - t0,
                    )
            finally:
                shutil.rmtree(stage_root, ignore_errors=True)

        sync_thread = threading.Thread(
            target=_sync_staged_to_target,
            name=f"miles-ckpt-sync-s{step_id}-r{rank}",
            daemon=False,
        )
        actor._temp_stage_sync_thread = sync_thread
        _ACTIVE_SYNC_THREADS.append(sync_thread)
        sync_thread.start()
        if force_sync:
            flush_staged_saves(actor)
            dist.barrier(group=gloo_group)

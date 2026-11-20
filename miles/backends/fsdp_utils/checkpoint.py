from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful

from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)
_CORE_CHECKPOINT_ENTRIES = frozenset({"model", "optimizer", "lr_scheduler", "rng.pt"})
_SYNC_WAIT_TIMEOUT_SECONDS = 3600.0
_SYNC_POLL_INTERVAL_SECONDS = 0.2


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
        tracker_text = tracker_file.read_text().strip() if tracker_file.exists() else ""
        target_step = int(tracker_text) if tracker_text.isdigit() else -1
        if dist.is_initialized():
            step_tensor = torch.tensor([target_step], dtype=torch.int64, device="cuda")
            dist.all_reduce(step_tensor, op=dist.ReduceOp.MAX)
            target_step = int(step_tensor.item())
        if target_step < 0:
            logger.info(f"[FSDP] No tracker file at {tracker_file}; skipping load.")
            return None

    checkpoint_dir = root_path / f"iter_{target_step:07d}"
    model_dir = checkpoint_dir / "model"
    optimizer_dir = checkpoint_dir / "optimizer"
    lr_scheduler_dir = checkpoint_dir / "lr_scheduler"

    if not model_dir.exists():
        raise FileNotFoundError(f"[FSDP] Model checkpoint {model_dir} not found for tracked step {target_step}")

    gloo_group = get_gloo_group()

    model_state = ModelState(actor.model)
    dcp.load(state_dict={"model_state": model_state}, checkpoint_id=str(model_dir), process_group=gloo_group)
    logger.info(f"[FSDP] Loaded model from {model_dir}")

    load_optimizer = not getattr(actor.args, "no_load_optim", False) and hasattr(actor, "optimizer")
    if load_optimizer and optimizer_dir.exists():
        optimizer_state = OptimizerState(actor.model, actor.optimizer)
        dcp.load(
            state_dict={"optim_state": optimizer_state},
            checkpoint_id=str(optimizer_dir),
            process_group=gloo_group,
        )
        logger.info(f"[FSDP] Loaded optimizer from {optimizer_dir}")
    elif load_optimizer:
        logger.info(f"[FSDP] Optimizer checkpoint not found at {optimizer_dir}, skipping optimizer load.")

    load_lr_scheduler = (
        not getattr(actor.args, "no_load_optim", False)
        and hasattr(actor, "lr_scheduler")
        and lr_scheduler_dir.exists()
    )
    if load_lr_scheduler:
        lr_scheduler_state = LRSchedulerState(actor.lr_scheduler)
        dcp.load(
            state_dict={"lr_scheduler_state": lr_scheduler_state},
            checkpoint_id=str(lr_scheduler_dir),
            process_group=gloo_group,
        )
        logger.info(f"[FSDP] Loaded LR scheduler from {lr_scheduler_dir}")
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


atexit.register(flush_staged_saves)


def _checkpoint_sync_sort_key(rel_path: Path) -> tuple[int, str]:
    top_name = rel_path.parts[0]
    if top_name in _CORE_CHECKPOINT_ENTRIES:
        tier = 0
    elif top_name == "meta.json":
        tier = 2
    else:
        tier = 1
    return tier, str(rel_path)


def _copy_staged_entry(src_item: Path, dst_item: Path, rank: int) -> None:
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


def _wait_for_all_ranks_synced(sync_done_dir: Path, world_size: int, timeout_sec: float = _SYNC_WAIT_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + timeout_sec
    expected = {f"rank_{r:05d}.ok" for r in range(world_size)}
    while time.monotonic() < deadline:
        entries = {p.name for p in sync_done_dir.iterdir()} if sync_done_dir.is_dir() else set()
        errors = sorted(name for name in entries if name.endswith(".err"))
        if errors:
            detail = (sync_done_dir / errors[0]).read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"[FSDP] Staged checkpoint sync failed on peer ({errors[0]}): {detail}")
        if expected.issubset(entries):
            shutil.rmtree(sync_done_dir, ignore_errors=True)
            return
        time.sleep(_SYNC_POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"[FSDP] Timed out waiting for {world_size} ranks to finish checkpoint sync in {sync_done_dir}")


def _sync_staged_checkpoint(
    *,
    stage_root: Path,
    write_ckpt_dir: Path,
    checkpoint_dir: Path,
    base_dir: Path,
    sync_done_dir: Path,
    step_id: int,
    rank: int,
    world_size: int,
) -> None:
    t0 = time.perf_counter()
    try:
        ordered_items = sorted(
            write_ckpt_dir.rglob("*"),
            key=lambda p: _checkpoint_sync_sort_key(p.relative_to(write_ckpt_dir)),
        )
        for src_item in ordered_items:
            _copy_staged_entry(src_item, checkpoint_dir / src_item.relative_to(write_ckpt_dir), rank)
        done_tmp = sync_done_dir / f"rank_{rank:05d}.ok.tmp"
        done_tmp.write_text("ok", encoding="utf-8")
        done_tmp.replace(sync_done_dir / f"rank_{rank:05d}.ok")
        if rank == 0:
            _wait_for_all_ranks_synced(sync_done_dir, world_size)
            tracker_tmp = base_dir / f"latest_checkpointed_iteration.txt.tmp.s{step_id}"
            tracker_tmp.write_text(str(step_id), encoding="utf-8")
            tracker_tmp.replace(base_dir / "latest_checkpointed_iteration.txt")
            logger.info(
                "[FSDP] Synced staged checkpoint to %s across %d ranks in %.2fs",
                checkpoint_dir,
                world_size,
                time.perf_counter() - t0,
            )
    except Exception as exc:
        try:
            sync_done_dir.mkdir(parents=True, exist_ok=True)
            (sync_done_dir / f"rank_{rank:05d}.err").write_text(repr(exc), encoding="utf-8")
        except OSError:
            pass
        raise
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)


def save(actor: Any, iteration: int, force_sync: bool = False) -> None:
    """Save checkpoint to disk with local staging and cross-rank atomic tracker commit."""
    torch.cuda.synchronize()
    flush_staged_saves(actor)
    gloo_group = get_gloo_group()

    base_dir = Path(actor.args.save).expanduser()
    step_id = iteration + 1
    checkpoint_dir = base_dir / f"iter_{step_id:07d}"
    rank = dist.get_rank()
    world_size = dist.get_world_size()
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

    dcp.save({"model_state": ModelState(actor.model)}, checkpoint_id=str(model_dir))

    save_optimizer_state = not getattr(actor.args, "no_save_optim", False)
    if save_optimizer_state and hasattr(actor, "optimizer") and actor.optimizer is not None:
        dcp.save(
            {"optim_state": OptimizerState(actor.model, actor.optimizer)},
            checkpoint_id=str(optimizer_dir),
        )
    if save_optimizer_state and hasattr(actor, "lr_scheduler") and actor.lr_scheduler is not None:
        dcp.save(
            {"lr_scheduler_state": LRSchedulerState(actor.lr_scheduler)},
            checkpoint_id=str(lr_scheduler_dir),
        )

    if rank == 0:
        rng_state = {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
        torch.save(rng_state, write_ckpt_dir / "rng.pt")
        metadata = {
            "iteration": step_id,
            "rollout_id": iteration,
            "next_rollout_id": iteration + 1,
            "global_step": actor.global_step,
            "micro_step": actor.micro_step,
            "world_size": world_size,
            "timestamp": time.time(),
        }
        _write_checkpoint_metadata(write_ckpt_dir / "meta.json", metadata)

    dist.barrier(group=gloo_group)

    if not temp_stage_save:
        if rank == 0:
            tracker_tmp = base_dir / f"latest_checkpointed_iteration.txt.tmp.s{step_id}"
            tracker_tmp.write_text(str(step_id), encoding="utf-8")
            tracker_tmp.replace(base_dir / "latest_checkpointed_iteration.txt")
            logger.info(f"[FSDP] Saved checkpoint to {checkpoint_dir}")
        dist.barrier(group=gloo_group)
        return

    assert stage_root is not None
    sync_done_dir = checkpoint_dir / ".sync_done"
    if rank == 0:
        shutil.rmtree(sync_done_dir, ignore_errors=True)
        sync_done_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("model", "optimizer", "lr_scheduler"):
            (checkpoint_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.info(f"[FSDP] Staged checkpoint to local {write_ckpt_dir}; starting sync to {checkpoint_dir}")
    dist.barrier(group=gloo_group)

    sync_thread = threading.Thread(
        target=_sync_staged_checkpoint,
        kwargs={
            "stage_root": stage_root,
            "write_ckpt_dir": write_ckpt_dir,
            "checkpoint_dir": checkpoint_dir,
            "base_dir": base_dir,
            "sync_done_dir": sync_done_dir,
            "step_id": step_id,
            "rank": rank,
            "world_size": world_size,
        },
        name=f"miles-ckpt-sync-s{step_id}-r{rank}",
        daemon=False,
    )
    actor._temp_stage_sync_thread = sync_thread
    _ACTIVE_SYNC_THREADS.append(sync_thread)
    sync_thread.start()
    if force_sync:
        flush_staged_saves(actor)
        dist.barrier(group=gloo_group)

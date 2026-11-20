"""HBM3-resident hot keyword feature cache backed by /dev/shm precompiled table."""

from __future__ import annotations

import os
from pathlib import Path

from filelock import FileLock
import numpy as np
import torch

_BYTES_PER_GIB = 1024**3
_RING_FRACTION = 0.125
_LOAD_CHUNK_ROWS = 4_194_304
_MIN_CACHE_SLOTS = 1_048_576
_MAX_RING_SLOTS = 16_777_216


def load_or_build_degree_order(
    num_keywords: int,
    kw_degrees: np.ndarray | None,
    tag: str,
) -> np.ndarray | None:
    if kw_degrees is None or len(kw_degrees) != num_keywords:
        return None
    shm_root = Path("/dev/shm")
    if not shm_root.is_dir():
        return np.argsort(-kw_degrees, kind="quicksort").astype(np.int32)
    order_path = shm_root / f"gr_kw_deg_order_{tag}.i32.bin"
    expected_bytes = num_keywords * 4
    with FileLock(str(shm_root / f"gr_kw_deg_order_{tag}.lock")):
        if not (order_path.is_file() and order_path.stat().st_size == expected_bytes):
            order_i32 = np.argsort(-kw_degrees, kind="quicksort").astype(np.int32)
            tmp_path = order_path.with_suffix(f".tmp.{os.getpid()}")
            try:
                tmp_path.write_bytes(order_i32.tobytes())
                tmp_path.replace(order_path)
            finally:
                tmp_path.unlink(missing_ok=True)
    return np.memmap(order_path, dtype=np.int32, mode="r", shape=(num_keywords,))


def resolve_hbm_budget_bytes(
    device: torch.device,
    *,
    num_keywords: int,
    enabled: bool = True,
    reserve_gib: float = 82.0,
    max_gib: float = 60.0,
    explicit_gib: float | None = None,
) -> int:
    if not enabled or device.type != "cuda":
        return 0
    index_bytes = num_keywords * 4
    if explicit_gib is not None:
        return max(0, int(explicit_gib * _BYTES_PER_GIB) - index_bytes)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    usable = min(int(max_gib * _BYTES_PER_GIB), int(free_bytes - reserve_gib * _BYTES_PER_GIB)) - index_bytes
    granularity = 2 * _BYTES_PER_GIB
    if usable < granularity:
        return 0
    return (usable // granularity) * granularity


class HbmKeywordCache:
    """Two-tier GPU HBM3 static-hot + FIFO-ring cache over a /dev/shm bf16 keyword table."""

    def __init__(
        self,
        *,
        mmap_u16: np.memmap,
        num_keywords: int,
        num_tokens: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        kw_degrees: np.ndarray | None = None,
        tag: str = "part25",
        enabled: bool = True,
        reserve_gib: float = 82.0,
        max_gib: float = 60.0,
        explicit_gib: float | None = None,
    ) -> None:
        self._mmap = mmap_u16
        self.num_keywords = num_keywords
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim
        self.device = device
        self.dtype = dtype
        row_bytes = num_tokens * feature_dim * 2
        budget_bytes = resolve_hbm_budget_bytes(
            device,
            num_keywords=num_keywords,
            enabled=enabled,
            reserve_gib=reserve_gib,
            max_gib=max_gib,
            explicit_gib=explicit_gib,
        )
        total_slots = min(num_keywords, budget_bytes // row_bytes) if budget_bytes > 0 else 0

        self.enabled = total_slots >= _MIN_CACHE_SLOTS
        self.total_slots = total_slots if self.enabled else 0
        self.ring_slots = (
            min(_MAX_RING_SLOTS, int(self.total_slots * _RING_FRACTION))
            if self.total_slots < num_keywords
            else 0
        )
        self.static_slots = self.total_slots - self.ring_slots
        self._ring_cursor = 0
        self.hbm_hits = 0
        self.shm_misses = 0

        if not self.enabled:
            self._kid_to_slot: torch.Tensor | None = None
            self._hbm_table: torch.Tensor | None = None
            self._ring_owners: torch.Tensor | None = None
            return

        self._kid_to_slot = torch.full((num_keywords,), -1, dtype=torch.int32, device=device)
        self._hbm_table = torch.empty((self.total_slots, num_tokens, feature_dim), dtype=dtype, device=device)
        self._ring_owners = (
            torch.full((self.ring_slots,), -1, dtype=torch.int64, device=device)
            if self.ring_slots > 0
            else None
        )
        self._populate_static_slots(kw_degrees, tag)

    def _populate_static_slots(self, kw_degrees: np.ndarray | None, tag: str) -> None:
        assert self._kid_to_slot is not None and self._hbm_table is not None
        order = load_or_build_degree_order(self.num_keywords, kw_degrees, tag)
        hot_kids = (
            np.sort(np.asarray(order[: self.static_slots], dtype=np.int64))
            if order is not None
            else np.arange(self.static_slots, dtype=np.int64)
        )
        for start in range(0, self.static_slots, _LOAD_CHUNK_ROWS):
            end = min(start + _LOAD_CHUNK_ROWS, self.static_slots)
            chunk_kids = hot_kids[start:end]
            cpu_u16 = np.array(self._mmap[chunk_kids], copy=True)
            t_bf16 = torch.from_numpy(cpu_u16).view(torch.bfloat16)
            self._hbm_table[start:end].copy_(t_bf16.to(dtype=self.dtype), non_blocking=False)
            kids_gpu = torch.from_numpy(chunk_kids).to(device=self.device)
            slots_gpu = torch.arange(start, end, dtype=torch.int32, device=self.device)
            self._kid_to_slot[kids_gpu] = slots_gpu

    def _gather_from_shm(self, kids_cpu: np.ndarray) -> torch.Tensor:
        u16_copy = np.array(self._mmap[kids_cpu], copy=True)
        cpu_tensor = torch.from_numpy(u16_copy).view(torch.bfloat16)
        if self.device.type == "cuda":
            cpu_tensor = cpu_tensor.pin_memory()
        return cpu_tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def gather(self, unique_kids_gpu: torch.Tensor) -> torch.Tensor:
        n = int(unique_kids_gpu.numel())
        if n == 0:
            return torch.empty((0, self.num_tokens, self.feature_dim), device=self.device, dtype=self.dtype)
        if not self.enabled or self._kid_to_slot is None and self._hbm_table is None:
            self.shm_misses += n
            return self._gather_from_shm(unique_kids_gpu.detach().cpu().numpy())
        assert self._kid_to_slot is not None and self._hbm_table is not None

        slots = self._kid_to_slot[unique_kids_gpu]
        hit_mask = slots >= 0
        if bool(hit_mask.all().item()):
            self.hbm_hits += n
            return self._hbm_table[slots.long()]

        out = torch.empty((n, self.num_tokens, self.feature_dim), device=self.device, dtype=self.dtype)
        hit_idx = torch.nonzero(hit_mask, as_tuple=False).squeeze(-1)
        miss_idx = torch.nonzero(~hit_mask, as_tuple=False).squeeze(-1)
        n_hit = int(hit_idx.numel())
        n_miss = int(miss_idx.numel())
        self.hbm_hits += n_hit
        self.shm_misses += n_miss
        if n_hit > 0:
            out[hit_idx] = self._hbm_table[slots[hit_idx].long()]

        miss_kids_gpu = unique_kids_gpu[miss_idx]
        miss_feats = self._gather_from_shm(miss_kids_gpu.detach().cpu().numpy())
        out[miss_idx] = miss_feats

        if self.ring_slots > 0 and self._ring_owners is not None:
            ins = min(n_miss, self.ring_slots)
            ring_pos = (torch.arange(ins, device=self.device, dtype=torch.int64) + self._ring_cursor) % self.ring_slots
            self._ring_cursor = int((self._ring_cursor + ins) % self.ring_slots)
            old_owners = self._ring_owners[ring_pos]
            valid_old = old_owners >= 0
            if bool(valid_old.any().item()):
                self._kid_to_slot[old_owners[valid_old]] = -1
            table_slots = ring_pos + self.static_slots
            ins_kids = miss_kids_gpu[:ins]
            self._hbm_table[table_slots] = miss_feats[:ins]
            self._ring_owners[ring_pos] = ins_kids
            self._kid_to_slot[ins_kids] = table_slots.to(torch.int32)

        return out

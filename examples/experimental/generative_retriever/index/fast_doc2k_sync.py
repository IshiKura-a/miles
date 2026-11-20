"""Append-only per-microbatch doc2k delta staging and rank-sharded vectorized vote sync."""

from __future__ import annotations

from pathlib import Path
import pickle
import struct
import tempfile
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from examples.experimental.generative_retriever.index.doc2k_voting import Doc2kVoteDelta
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DOC2K_ADD_MIN_QUERIES,
    DOC2K_PRUNE_FLOOR,
    DOC2K_PRUNE_MIN_QUERIES,
)
from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping

_HEADER_STRUCT = struct.Struct("<III")
_SIGN_FLIP_U64 = np.uint64(1 << 63)


def local_doc2k_delta_path(local_gpu_id: int) -> Path:
    return Path(tempfile.gettempdir()) / f"gr_doc2k_local_gpu{local_gpu_id}.delta.bin"


def global_doc2k_delta_path(local_gpu_id: int) -> Path:
    return Path(tempfile.gettempdir()) / f"gr_doc2k_global_gpu{local_gpu_id}.pkl"


def append_local_doc2k_delta(local_gpu_id: int, delta: Doc2kVoteDelta) -> None:
    if delta.is_empty():
        return
    n_cand = len(delta.cand_votes)
    n_docq = len(delta.docid_q_count)
    n_edge = len(delta.edge_passes)

    parts: list[bytes] = [_HEADER_STRUCT.pack(n_cand, n_docq, n_edge)]
    if n_cand > 0:
        ck = np.fromiter(
            ((np.uint64(u32) << np.uint64(32)) | np.uint64(kid) for (u32, kid) in delta.cand_votes),
            dtype=np.uint64,
            count=n_cand,
        )
        cv = np.fromiter(delta.cand_votes.values(), dtype=np.int32, count=n_cand)
        parts.extend((ck.tobytes(), cv.tobytes()))
    if n_docq > 0:
        dk = np.fromiter(delta.docid_q_count.keys(), dtype=np.uint32, count=n_docq)
        dv = np.fromiter(delta.docid_q_count.values(), dtype=np.int32, count=n_docq)
        parts.extend((dk.tobytes(), dv.tobytes()))
    if n_edge > 0:
        ek = np.fromiter(
            ((np.uint64(u32) << np.uint64(32)) | np.uint64(kid) for (u32, kid) in delta.edge_passes),
            dtype=np.uint64,
            count=n_edge,
        )
        ev = np.fromiter(delta.edge_passes.values(), dtype=np.int32, count=n_edge)
        parts.extend((ek.tobytes(), ev.tobytes()))

    with open(local_doc2k_delta_path(local_gpu_id), "ab") as stream:
        stream.write(b"".join(parts))


def consume_local_doc2k_delta(
    local_gpu_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = local_doc2k_delta_path(local_gpu_id)
    empty_u64 = np.empty(0, dtype=np.uint64)
    empty_u32 = np.empty(0, dtype=np.uint32)
    empty_i32 = np.empty(0, dtype=np.int32)
    if not path.is_file():
        return empty_u64, empty_i32, empty_u32, empty_i32, empty_u64, empty_i32
    try:
        raw = path.read_bytes()
    finally:
        path.unlink(missing_ok=True)
    if not raw:
        return empty_u64, empty_i32, empty_u32, empty_i32, empty_u64, empty_i32

    ck_list: list[np.ndarray] = []
    cv_list: list[np.ndarray] = []
    dk_list: list[np.ndarray] = []
    dv_list: list[np.ndarray] = []
    ek_list: list[np.ndarray] = []
    ev_list: list[np.ndarray] = []

    offset = 0
    total = len(raw)
    hdr_sz = _HEADER_STRUCT.size
    while offset < total:
        n_cand, n_docq, n_edge = _HEADER_STRUCT.unpack_from(raw, offset)
        offset += hdr_sz
        if n_cand > 0:
            ck_list.append(np.frombuffer(raw, dtype=np.uint64, count=n_cand, offset=offset))
            offset += n_cand * 8
            cv_list.append(np.frombuffer(raw, dtype=np.int32, count=n_cand, offset=offset))
            offset += n_cand * 4
        if n_docq > 0:
            dk_list.append(np.frombuffer(raw, dtype=np.uint32, count=n_docq, offset=offset))
            offset += n_docq * 4
            dv_list.append(np.frombuffer(raw, dtype=np.int32, count=n_docq, offset=offset))
            offset += n_docq * 4
        if n_edge > 0:
            ek_list.append(np.frombuffer(raw, dtype=np.uint64, count=n_edge, offset=offset))
            offset += n_edge * 8
            ev_list.append(np.frombuffer(raw, dtype=np.int32, count=n_edge, offset=offset))
            offset += n_edge * 4

    return (
        np.concatenate(ck_list).copy() if ck_list else empty_u64,
        np.concatenate(cv_list).copy() if cv_list else empty_i32,
        np.concatenate(dk_list).copy() if dk_list else empty_u32,
        np.concatenate(dv_list).copy() if dv_list else empty_i32,
        np.concatenate(ek_list).copy() if ek_list else empty_u64,
        np.concatenate(ev_list).copy() if ev_list else empty_i32,
    )


def merge_and_sum_u64(
    keys_a: np.ndarray,
    vals_a: np.ndarray,
    keys_b: np.ndarray,
    vals_b: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(keys_b) == 0:
        return keys_a, vals_a
    all_k = np.concatenate((keys_a, keys_b)) if len(keys_a) > 0 else keys_b
    all_v = np.concatenate((vals_a, vals_b)) if len(vals_a) > 0 else vals_b
    if len(all_k) == 0:
        return all_k, all_v

    if device is not None and device.type == "cuda" and len(all_k) >= 4096:
        signed_k = (all_k ^ _SIGN_FLIP_U64).view(np.int64)
        k_t = torch.from_numpy(signed_k).to(device=device)
        v_t = torch.from_numpy(all_v.astype(np.int64)).to(device=device)
        order = torch.argsort(k_t)
        uniq_t, inv_t = torch.unique_consecutive(k_t[order], return_inverse=True)
        sum_v_t = torch.zeros(len(uniq_t), dtype=torch.int64, device=device)
        sum_v_t.scatter_add_(0, inv_t, v_t[order])
        uniq_u64 = uniq_t.cpu().numpy().view(np.uint64) ^ _SIGN_FLIP_U64
        return uniq_u64, sum_v_t.cpu().numpy().astype(np.int32)

    order_np = np.argsort(all_k, kind="mergesort")
    sk = all_k[order_np]
    sv = all_v[order_np].astype(np.int64)
    uniq, first_idx = np.unique(sk, return_index=True)
    return uniq, np.add.reduceat(sv, first_idx).astype(np.int32)


def _all_to_all_kv_u64(
    keys_u64: np.ndarray,
    vals_i32: np.ndarray,
    owners: np.ndarray,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> tuple[np.ndarray, np.ndarray]:
    if world_size == 1:
        return keys_u64, vals_i32
    if len(keys_u64) > 0:
        perm = np.argsort(owners, kind="mergesort")
        s_keys = np.ascontiguousarray(keys_u64[perm])
        s_vals = np.ascontiguousarray(vals_i32[perm])
        send_counts = np.bincount(owners[perm], minlength=world_size).astype(np.int64)
    else:
        s_keys = np.empty(0, dtype=np.uint64)
        s_vals = np.empty(0, dtype=np.int32)
        send_counts = np.zeros(world_size, dtype=np.int64)

    send_cnt_t = torch.from_numpy(send_counts)
    recv_cnt_t = torch.empty(world_size, dtype=torch.int64)
    dist.all_to_all_single(recv_cnt_t, send_cnt_t, group=group)
    recv_counts = recv_cnt_t.numpy()
    total_recv = int(recv_counts.sum())
    in_splits = send_counts.tolist()
    out_splits = recv_counts.tolist()

    sk_t = torch.from_numpy(s_keys.view(np.int64))
    sv_t = torch.from_numpy(s_vals)
    rk_t = torch.empty(total_recv, dtype=torch.int64)
    rv_t = torch.empty(total_recv, dtype=torch.int32)
    dist.all_to_all_single(rk_t, sk_t, output_split_sizes=out_splits, input_split_sizes=in_splits, group=group)
    dist.all_to_all_single(rv_t, sv_t, output_split_sizes=out_splits, input_split_sizes=in_splits, group=group)
    return rk_t.numpy().view(np.uint64), rv_t.numpy()


def _all_gather_u64_to_all(
    arr_u64: np.ndarray,
    world_size: int,
    group: dist.ProcessGroup | None,
) -> np.ndarray:
    if world_size == 1:
        return arr_u64
    len_t = torch.tensor([len(arr_u64)], dtype=torch.int64)
    all_lens_t = torch.empty(world_size, dtype=torch.int64)
    dist.all_gather_into_tensor(all_lens_t, len_t, group=group)
    lens = all_lens_t.tolist()
    max_len = max(lens)
    if max_len == 0:
        return np.empty(0, dtype=np.uint64)
    send_t = torch.zeros(max_len, dtype=torch.int64)
    if len(arr_u64) > 0:
        send_t[: len(arr_u64)] = torch.from_numpy(np.ascontiguousarray(arr_u64).view(np.int64))
    recv_t = torch.empty(world_size * max_len, dtype=torch.int64)
    dist.all_gather_into_tensor(recv_t, send_t, group=group)
    recv_np = recv_t.numpy().view(np.uint64)
    chunks = [recv_np[r * max_len : r * max_len + r_len] for r, r_len in enumerate(lens) if r_len > 0]
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)


def _build_global_mapping_delta(
    sync_map: ResidentMapping | None,
    all_prom_u64: np.ndarray,
    all_harv_u64: np.ndarray,
    all_edges_u64: np.ndarray,
) -> dict[str, Any]:
    additions: dict[int, list[int]] = {}
    for packed in np.sort(all_prom_u64).tolist():
        target_u32 = int(packed >> 32)
        kid = int(packed & 0xFFFFFFFF)
        bucket_list = additions.setdefault(target_u32, [])
        if kid not in bucket_list:
            bucket_list.append(kid)

    passed_by_u32: dict[int, set[int]] = {}
    for packed_e in all_edges_u64.tolist():
        passed_by_u32.setdefault(int(packed_e >> 32), set()).add(int(packed_e & 0xFFFFFFFF))

    removals: dict[int, set[int]] = {}
    for u32_raw in np.sort(all_harv_u64).tolist():
        u32 = int(u32_raw)
        kids_tup = sync_map.get_docid_kids(u32) if sync_map is not None else ()
        if len(kids_tup) <= DOC2K_PRUNE_FLOOR:
            continue
        passed_set = passed_by_u32.get(u32)
        dead_kids = set(kids_tup) if passed_set is None else {kid for kid in kids_tup if kid not in passed_set}
        if dead_kids:
            removals[u32] = dead_kids

    if sync_map is not None:
        sync_map.apply_u32_kid_delta(additions, removals)
    return {"additions": additions, "removals": removals}


def _stage_global_delta_for_local_gpu(local_gpu: int, delta: dict[str, Any]) -> None:
    additions_local = {u32: list(kids) for u32, kids in delta["additions"].items()}
    removals_local = {u32: set(kids) for u32, kids in delta["removals"].items()}
    global_path = global_doc2k_delta_path(local_gpu)
    if global_path.is_file():
        prev = pickle.loads(global_path.read_bytes())
        for u32, kids in prev.get("additions", {}).items():
            bl = additions_local.setdefault(u32, [])
            for k in kids:
                if k not in bl:
                    bl.append(k)
        for u32, kids in prev.get("removals", {}).items():
            removals_local.setdefault(u32, set()).update(kids)
    tmp_path = global_path.with_suffix(".tmp")
    tmp_path.write_bytes(
        pickle.dumps({"additions": additions_local, "removals": removals_local}, protocol=pickle.HIGHEST_PROTOCOL)
    )
    tmp_path.replace(global_path)


class FastDoc2kSyncState:
    """Rank-sharded flat C-array state for lossless, sub-second global doc2k vote sync."""

    def __init__(self) -> None:
        self.cand_keys_u64 = np.empty(0, dtype=np.uint64)
        self.cand_cnts_i32 = np.empty(0, dtype=np.int32)
        self.docq_keys_u64 = np.empty(0, dtype=np.uint64)
        self.docq_cnts_i32 = np.empty(0, dtype=np.int32)
        self.edge_keys_u64 = np.empty(0, dtype=np.uint64)
        self.edge_cnts_i32 = np.empty(0, dtype=np.int32)
        self._cpu_group: dist.ProcessGroup | None = None

    def _ensure_cpu_group(self, world_size: int) -> dist.ProcessGroup | None:
        if world_size <= 1:
            return None
        if self._cpu_group is None:
            pg = dist.new_group(backend="gloo")
            assert isinstance(pg, dist.ProcessGroup)
            self._cpu_group = pg
        return self._cpu_group

    def harvest_local_shard(
        self,
        in_ck: np.ndarray,
        in_cv: np.ndarray,
        in_dk_u64: np.ndarray,
        in_dv: np.ndarray,
        in_ek: np.ndarray,
        in_ev: np.ndarray,
        device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.cand_keys_u64, self.cand_cnts_i32 = merge_and_sum_u64(
            self.cand_keys_u64, self.cand_cnts_i32, in_ck, in_cv, device=device
        )
        prom_mask = self.cand_cnts_i32 >= DOC2K_ADD_MIN_QUERIES
        prom_u64 = self.cand_keys_u64[prom_mask]
        if prom_mask.any():
            keep_c = ~prom_mask
            self.cand_keys_u64 = self.cand_keys_u64[keep_c]
            self.cand_cnts_i32 = self.cand_cnts_i32[keep_c]

        self.docq_keys_u64, self.docq_cnts_i32 = merge_and_sum_u64(
            self.docq_keys_u64, self.docq_cnts_i32, in_dk_u64, in_dv, device=device
        )
        self.edge_keys_u64, self.edge_cnts_i32 = merge_and_sum_u64(
            self.edge_keys_u64, self.edge_cnts_i32, in_ek, in_ev, device=device
        )

        harv_mask = self.docq_cnts_i32 >= DOC2K_PRUNE_MIN_QUERIES
        harv_u64 = self.docq_keys_u64[harv_mask]
        harv_passed_edges_u64 = np.empty(0, dtype=np.uint64)
        if harv_mask.any():
            keep_d = ~harv_mask
            self.docq_keys_u64 = self.docq_keys_u64[keep_d]
            self.docq_cnts_i32 = self.docq_cnts_i32[keep_d]
            if len(self.edge_keys_u64) > 0:
                edge_doc_u64 = (self.edge_keys_u64 >> np.uint64(32)).astype(np.uint64, copy=False)
                pos = np.minimum(np.searchsorted(harv_u64, edge_doc_u64), len(harv_u64) - 1)
                in_harv = harv_u64[pos] == edge_doc_u64
                if in_harv.any():
                    harv_passed_edges_u64 = self.edge_keys_u64[in_harv & (self.edge_cnts_i32 > 0)]
                    keep_e = ~in_harv
                    self.edge_keys_u64 = self.edge_keys_u64[keep_e]
                    self.edge_cnts_i32 = self.edge_cnts_i32[keep_e]

        return prom_u64, harv_u64, harv_passed_edges_u64

    def sync_distributed(
        self,
        *,
        local_gpu: int,
        rank: int,
        world_size: int,
        device: torch.device,
        ensure_sync_map_fn: Any,
    ) -> None:
        del device
        cpu_group = self._ensure_cpu_group(world_size)
        ck, cv, dk_u32, dv, ek, ev = consume_local_doc2k_delta(local_gpu)
        dk_u64 = dk_u32.astype(np.uint64)
        has_any_t = torch.tensor(
            [int(len(ck) > 0 or len(dk_u64) > 0 or len(ek) > 0)],
            dtype=torch.int64,
        )
        if world_size > 1:
            dist.all_reduce(has_any_t, op=dist.ReduceOp.SUM, group=cpu_group)
        if int(has_any_t.item()) == 0:
            return

        r_ck, r_cv = _all_to_all_kv_u64(
            ck, cv, ((ck >> np.uint64(32)) % np.uint64(world_size)).astype(np.int64), world_size, cpu_group
        )
        r_dk, r_dv = _all_to_all_kv_u64(
            dk_u64, dv, (dk_u64 % np.uint64(world_size)).astype(np.int64), world_size, cpu_group
        )
        r_ek, r_ev = _all_to_all_kv_u64(
            ek, ev, ((ek >> np.uint64(32)) % np.uint64(world_size)).astype(np.int64), world_size, cpu_group
        )

        prom_u64, harv_u64, harv_edges_u64 = self.harvest_local_shard(
            r_ck, r_cv, r_dk, r_dv, r_ek, r_ev, device=None
        )
        all_prom_u64 = _all_gather_u64_to_all(prom_u64, world_size, cpu_group)
        all_harv_u64 = _all_gather_u64_to_all(harv_u64, world_size, cpu_group)
        all_edges_u64 = _all_gather_u64_to_all(harv_edges_u64, world_size, cpu_group)

        delta_bytes = (
            pickle.dumps(
                _build_global_mapping_delta(ensure_sync_map_fn(), all_prom_u64, all_harv_u64, all_edges_u64),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            if rank == 0
            else b""
        )
        delta_len_t = torch.tensor([len(delta_bytes)], dtype=torch.int64)
        if world_size > 1:
            dist.broadcast(delta_len_t, src=0, group=cpu_group)
        delta_len = int(delta_len_t.item())
        if delta_len == 0:
            return
        delta_buf = (
            torch.frombuffer(bytearray(delta_bytes), dtype=torch.uint8)
            if rank == 0
            else torch.empty(delta_len, dtype=torch.uint8)
        )
        if world_size > 1:
            dist.broadcast(delta_buf, src=0, group=cpu_group)
        _stage_global_delta_for_local_gpu(local_gpu, pickle.loads(bytes(delta_buf.numpy())))

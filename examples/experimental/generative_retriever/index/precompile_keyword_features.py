"""Precompile static Graphene keyword features for all unique keywords in a columnar GR index.

Writes per-node binary shards (``bfloat16`` viewed as ``uint16``, shape ``[N, 2, 128]``) directly to
local NVMe first, verifies numerical equivalence against ``RelevanceScorer._encode_gpu_table``,
and assembles ``keyword_features_gv21_large.bf16.bin`` + ``.meta.json``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from examples.experimental.generative_retriever.reward.precompiled_keywords import (
    PRECOMPILED_KW_BIN_NAME,
    PRECOMPILED_KW_META_NAME,
)
from examples.experimental.generative_retriever.reward.relevance_model import (
    EXPECTED_GRAPHENE_COMMIT,
    GRAPHENE_FEATURE_DIM,
    GRAPHENE_NUM_TOKENS,
    RelevanceScorer,
    load_relevance_pipeline_factory,
    optimize_graphene_pipeline,
    relevance_source_path,
    stage_local_relevance_model,
    stage_local_relevance_source,
)
from examples.experimental.generative_retriever.services.mapping_staging import (
    stage_local_mapping_dir,
)

logger = logging.getLogger(__name__)
_KEYWORD_PREFIX = "Document: "
_PAD_MULTIPLE = 256


def _as_utf8_col(col: pa.ChunkedArray) -> pa.ChunkedArray:
    if pa.types.is_binary(col.type) or pa.types.is_large_binary(col.type):
        target = pa.large_string() if pa.types.is_large_binary(col.type) else pa.string()
        return col.cast(target, safe=False)
    return col


def load_keyword_table(staged_map_dir: Path) -> pa.ChunkedArray:
    arrow_path = staged_map_dir / "keywords_only.arrow"
    if arrow_path.is_file():
        mmap_src = pa.memory_map(str(arrow_path), "r")
        tbl = pa.ipc.open_file(mmap_src).read_all()
        return _as_utf8_col(tbl.column("keyword"))
    tbl = pq.read_table(staged_map_dir / "keywords.parquet", columns=["keyword"])
    return _as_utf8_col(tbl.column("keyword"))


def prepare_keyword_batch_cpu(
    tokenizer: Any,
    max_len: int,
    pad_id: int,
    add_emb: bool,
    raw_keywords: list[str],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    orig_n = len(raw_keywords)
    rem = (-orig_n) % _PAD_MULTIPLE
    padded = raw_keywords + ([raw_keywords[-1]] * rem if rem else [])
    prompts = [_KEYWORD_PREFIX + kw for kw in padded]
    tokenized = tokenizer(prompts, padding=False, truncation=True, max_length=max_len)["input_ids"]
    n = len(tokenized)
    ids_np = np.full((n, max_len), pad_id, dtype=np.int64)
    att_np = np.zeros((n, max_len), dtype=np.int64)
    for i, row in enumerate(tokenized):
        if add_emb and row and row[-1] != pad_id:
            row = list(row[: max_len - 1]) + [pad_id]
        else:
            row = row[:max_len]
        k = len(row)
        if k > 0:
            arr = np.asarray(row, dtype=np.int64)
            ids_np[i, max_len - k :] = arr
            att_np[i, max_len - k :] = (arr != pad_id).astype(np.int64)
        if add_emb:
            att_np[i, -1] = 1
    return torch.from_numpy(ids_np).pin_memory(), torch.from_numpy(att_np).pin_memory(), orig_n


def _start_batch_producer(
    kw_col: pa.ChunkedArray,
    batch_ranges: list[tuple[int, int]],
    tokenizer: Any,
    max_len: int,
    pad_id: int,
    add_emb: bool,
    prep_threads: int,
) -> queue.Queue[tuple[int, int, torch.Tensor, torch.Tensor, int] | BaseException | None]:
    q_ready: queue.Queue[tuple[int, int, torch.Tensor, torch.Tensor, int] | BaseException | None] = queue.Queue(
        maxsize=max(4, prep_threads * 2)
    )

    def _prep_one(rng: tuple[int, int]) -> tuple[int, int, torch.Tensor, torch.Tensor, int]:
        b_start, b_end = rng
        raw_kws: list[str] = kw_col.slice(b_start, b_end - b_start).to_pylist()
        ids_t, att_t, orig_n = prepare_keyword_batch_cpu(tokenizer, max_len, pad_id, add_emb, raw_kws)
        return b_start, b_end, ids_t, att_t, orig_n

    def _producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=prep_threads) as pool:
                for item in pool.map(_prep_one, batch_ranges):
                    q_ready.put(item)
            q_ready.put(None)
        except BaseException as exc:
            q_ready.put(exc)

    threading.Thread(target=_producer, daemon=True).start()
    return q_ready


def _gpu_shard_worker(
    gpu_id: int,
    staged_map_dir_str: str,
    staged_rel_model: str,
    staged_rel_src_str: str,
    shard_bin_path_str: str,
    shard_start: int,
    gpu_start: int,
    gpu_end: int,
    batch_size: int,
    prep_threads: int,
    status_dict: Any,
) -> None:
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        os.environ.pop(var, None)
    os.environ["ROCR_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    factory = load_relevance_pipeline_factory(Path(staged_rel_src_str))
    pipeline = factory(
        staged_rel_model,
        cup="large",
        device="cuda:0",
        dtype="bf16",
        batch_size=batch_size,
        cross_batch_size=16384,
        show_progress=False,
    )
    optimize_graphene_pipeline(pipeline)
    engine = pipeline.keyword_engine
    engine.batch_size = batch_size

    kw_col = load_keyword_table(Path(staged_map_dir_str))
    total_gpu_rows = gpu_end - gpu_start
    byte_offset = (gpu_start - shard_start) * GRAPHENE_NUM_TOKENS * GRAPHENE_FEATURE_DIM * 2
    out_mmap = np.memmap(
        shard_bin_path_str,
        dtype=np.uint16,
        mode="r+",
        offset=byte_offset,
        shape=(total_gpu_rows, GRAPHENE_NUM_TOKENS, GRAPHENE_FEATURE_DIM),
    )

    batch_ranges = [(pos, min(pos + batch_size, gpu_end)) for pos in range(gpu_start, gpu_end, batch_size)]
    q_ready = _start_batch_producer(
        kw_col,
        batch_ranges,
        engine.tokenizer,
        int(engine.max_length),
        int(engine.tokenizer.pad_token_id),
        bool(getattr(engine, "add_embedding_token", False)),
        prep_threads,
    )

    started = time.perf_counter()
    done_rows = 0
    step_idx = 0
    with torch.no_grad():
        while True:
            item = q_ready.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            b_start, _, ids_pin, att_pin, orig_n = item
            encoded = engine.model(
                ids_pin.to(device=device, non_blocking=True),
                att_pin.to(device=device, non_blocking=True),
            ).to(dtype=torch.bfloat16)[:orig_n]
            if encoded.shape != (orig_n, GRAPHENE_NUM_TOKENS, GRAPHENE_FEATURE_DIM):
                raise RuntimeError(f"Unexpected encoded shape: {encoded.shape}")
            rel_s = b_start - gpu_start
            out_mmap[rel_s : rel_s + orig_n] = encoded.contiguous().view(torch.uint16).cpu().numpy()
            done_rows += orig_n
            step_idx += 1
            if step_idx % 50 == 0 or done_rows == total_gpu_rows:
                elapsed = max(1e-6, time.perf_counter() - started)
                rate = done_rows / elapsed
                status_dict[gpu_id] = (done_rows, total_gpu_rows, rate, (total_gpu_rows - done_rows) / max(1e-6, rate))

    out_mmap.flush()
    elapsed = max(1e-6, time.perf_counter() - started)
    status_dict[gpu_id] = (total_gpu_rows, total_gpu_rows, total_gpu_rows / elapsed, 0.0)


def verify_equivalence(
    staged_map_dir: Path,
    staged_rel_model: str,
    staged_rel_src: Path,
    batch_size: int,
) -> None:
    kw_col = load_keyword_table(staged_map_dir)
    n_total = len(kw_col)
    sample_indices = sorted({
        *range(0, min(300, n_total)),
        *range(n_total // 2, min(n_total, n_total // 2 + 300)),
        *range(max(0, n_total - 300), n_total),
    })
    sample_kws = [kw_col[i].as_py() for i in sample_indices]

    factory = load_relevance_pipeline_factory(staged_rel_src)
    pipeline = factory(
        staged_rel_model,
        cup="large",
        device="cuda:0",
        dtype="bf16",
        batch_size=4096,
        cross_batch_size=16384,
        show_progress=False,
    )
    scorer = RelevanceScorer(
        pipeline,
        feature_cache_size=256,
        score_cache_size=256,
        pair_batch_size=4096,
        keyword_cache_size=4096,
    )
    scorer.node_keyword_cache = None

    with torch.no_grad():
        ref_storage, ref_map, _ = scorer._encode_gpu_table(sample_kws, query=False)
        ref_slots = torch.tensor([ref_map[k] for k in sample_kws], dtype=torch.long, device=ref_storage.device)
        ref_emb = ref_storage[ref_slots].clone()

        engine = pipeline.keyword_engine
        ids_pin, att_pin, orig_n = prepare_keyword_batch_cpu(
            engine.tokenizer,
            int(engine.max_length),
            int(engine.tokenizer.pad_token_id),
            bool(getattr(engine, "add_embedding_token", False)),
            sample_kws,
        )
        chunks = [
            engine.model(ids_pin[s : s + batch_size].to(ref_storage.device), att_pin[s : s + batch_size].to(ref_storage.device)).to(
                torch.bfloat16
            )
            for s in range(0, ids_pin.shape[0], batch_size)
        ]
        test_emb = torch.cat(chunks, dim=0)[:orig_n]

        emb_max_diff = float((ref_emb.float() - test_emb.float()).abs().max().item())
        test_queries = ["wireless earbuds", "cheap flights to tokyo", "running shoes men", "iphone 16 case"]
        q_storage, q_map, _ = scorer._encode_gpu_table(test_queries, query=True)
        q_slots = [q_map[q] for q in test_queries for _ in range(len(sample_kws))]
        k_slots_idx = list(range(len(sample_kws))) * len(test_queries)
        ref_scores = scorer._cross_score_gpu(q_storage, ref_emb, q_slots, k_slots_idx)
        test_scores = scorer._cross_score_gpu(q_storage, test_emb, q_slots, k_slots_idx)
        score_max_diff = float(np.abs(ref_scores - test_scores).max())

    logger.info(
        "[verify] samples=%d batch_size=%d emb_max_diff=%.6g cross_score_max_diff=%.6g",
        len(sample_kws),
        batch_size,
        emb_max_diff,
        score_max_diff,
    )
    if score_max_diff > 2e-3:
        raise RuntimeError(f"Verification failed: cross_score_max_diff={score_max_diff}")
    del scorer, pipeline
    torch.cuda.empty_cache()


def run_shard(args: argparse.Namespace) -> None:
    staged_map_dir = stage_local_mapping_dir(args.mapping_dir)
    staged_rel_src = stage_local_relevance_source(relevance_source_path(args.relevance_model))
    staged_rel_model = stage_local_relevance_model(args.relevance_model)

    if not args.skip_verify:
        verify_equivalence(staged_map_dir, staged_rel_model, staged_rel_src, args.batch_size)

    kw_col = load_keyword_table(staged_map_dir)
    total_rows = len(kw_col)
    del kw_col

    shard_start = (total_rows * args.shard_id) // args.num_shards
    shard_end = (total_rows * (args.shard_id + 1)) // args.num_shards
    shard_rows = shard_end - shard_start
    shard_bytes = shard_rows * GRAPHENE_NUM_TOKENS * GRAPHENE_FEATURE_DIM * 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{args.shard_id}_of_{args.num_shards}.bf16.bin"
    done_sentinel = out_dir / f"shard_{args.shard_id}_of_{args.num_shards}.done.json"
    with open(shard_path, "wb") as f:
        f.truncate(shard_bytes)

    gpu_ids = [int(x) for x in args.gpus.split(",") if x.strip()]
    n_gpus = len(gpu_ids)
    ctx = mp.get_context("spawn")
    status_dict = ctx.Manager().dict()
    procs: list[Any] = []
    t0 = time.perf_counter()

    for idx, gpu_id in enumerate(gpu_ids):
        g_start = shard_start + (shard_rows * idx) // n_gpus
        g_end = shard_start + (shard_rows * (idx + 1)) // n_gpus
        proc = ctx.Process(
            target=_gpu_shard_worker,
            args=(
                gpu_id,
                str(staged_map_dir),
                staged_rel_model,
                str(staged_rel_src),
                str(shard_path),
                shard_start,
                g_start,
                g_end,
                args.batch_size,
                args.prep_threads,
                status_dict,
            ),
        )
        proc.start()
        procs.append(proc)

    while any(p.is_alive() for p in procs):
        for p in procs:
            if p.exitcode not in (None, 0):
                for other in procs:
                    if other.is_alive():
                        other.terminate()
                raise RuntimeError(f"GPU worker exited with code {p.exitcode}")
        time.sleep(15.0)
        snap = dict(status_dict)
        if snap:
            done_sum = sum(v[0] for v in snap.values())
            rate_sum = sum(v[2] for v in snap.values())
            max_eta = max(v[3] for v in snap.values())
            logger.info(
                "[shard %d/%d] %d/%d (%.1f%%) node_rate=%.0f kw/s max_gpu_eta=%.0fs",
                args.shard_id,
                args.num_shards,
                done_sum,
                shard_rows,
                100.0 * done_sum / max(1, shard_rows),
                rate_sum,
                max_eta,
            )

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"GPU worker exited with code {p.exitcode}")

    elapsed = time.perf_counter() - t0
    meta = {
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_rows": total_rows,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "shard_rows": shard_rows,
        "shard_bytes": shard_bytes,
        "elapsed_seconds": round(elapsed, 2),
        "graphene_commit": EXPECTED_GRAPHENE_COMMIT,
    }
    done_sentinel.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_verify_assembled(args: argparse.Namespace) -> None:
    staged_map_dir = stage_local_mapping_dir(args.mapping_dir)
    staged_rel_src = stage_local_relevance_source(relevance_source_path(args.relevance_model))
    staged_rel_model = stage_local_relevance_model(args.relevance_model)

    kw_col = load_keyword_table(staged_map_dir)
    total_rows = len(kw_col)
    expected_bytes = total_rows * GRAPHENE_NUM_TOKENS * GRAPHENE_FEATURE_DIM * 2
    bin_path = Path(args.output_dir) / PRECOMPILED_KW_BIN_NAME
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(f"Byte size mismatch: got {actual_bytes}, expected {expected_bytes}")

    mmap_arr = np.memmap(
        bin_path,
        dtype=np.uint16,
        mode="r",
        shape=(total_rows, GRAPHENE_NUM_TOKENS, GRAPHENE_FEATURE_DIM),
    )

    probe_ids = [0, 1, total_rows - 1]
    for s in range(1, args.num_shards):
        boundary = (total_rows * s) // args.num_shards
        probe_ids.extend([boundary - 1, boundary, boundary + 1])
    rng = np.random.default_rng(42)
    probe_ids.extend(int(x) for x in rng.integers(0, total_rows, size=128))
    probe_ids = sorted(set(probe_ids))
    probe_kws = [kw_col[i].as_py() for i in probe_ids]

    factory = load_relevance_pipeline_factory(staged_rel_src)
    pipeline = factory(
        staged_rel_model,
        cup="large",
        device="cuda:0",
        dtype="bf16",
        batch_size=4096,
        cross_batch_size=16384,
        show_progress=False,
    )
    scorer = RelevanceScorer(
        pipeline,
        feature_cache_size=256,
        score_cache_size=256,
        pair_batch_size=4096,
        keyword_cache_size=4096,
    )
    scorer.node_keyword_cache = None
    ref_storage, ref_map, _ = scorer._encode_gpu_table(probe_kws, query=False)
    ref_slots = torch.tensor([ref_map[k] for k in probe_kws], dtype=torch.long, device=ref_storage.device)
    ref_emb = ref_storage[ref_slots].detach().cpu().float()

    loaded_emb = torch.from_numpy(np.array(mmap_arr[probe_ids], copy=True)).view(torch.bfloat16).float()
    max_diff = float((ref_emb - loaded_emb).abs().max().item())
    if not np.isfinite(loaded_emb.numpy()).all() or max_diff > 5e-2:
        raise RuntimeError(f"Assembled verification failed: max_diff={max_diff}")

    meta_payload = {
        "format_version": 1,
        "binary_file": PRECOMPILED_KW_BIN_NAME,
        "num_keywords": total_rows,
        "num_tokens": GRAPHENE_NUM_TOKENS,
        "feature_dim": GRAPHENE_FEATURE_DIM,
        "dtype": "bfloat16",
        "total_bytes": expected_bytes,
        "graphene_commit": EXPECTED_GRAPHENE_COMMIT,
        "relevance_model": str(Path(args.relevance_model).name),
        "mapping_name": str(Path(args.mapping_dir).name),
        "spot_check_probes": len(probe_ids),
        "spot_check_max_abs_diff": max_diff,
    }
    (Path(args.output_dir) / PRECOMPILED_KW_META_NAME).write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_shard = sub.add_parser("shard")
    p_shard.add_argument("--mapping-dir", required=True)
    p_shard.add_argument("--relevance-model", required=True)
    p_shard.add_argument("--output-dir", required=True)
    p_shard.add_argument("--shard-id", type=int, required=True)
    p_shard.add_argument("--num-shards", type=int, default=4)
    p_shard.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p_shard.add_argument("--batch-size", type=int, default=8192)
    p_shard.add_argument("--prep-threads", type=int, default=3)
    p_shard.add_argument("--skip-verify", action="store_true")

    p_ver = sub.add_parser("verify-assembled")
    p_ver.add_argument("--mapping-dir", required=True)
    p_ver.add_argument("--relevance-model", required=True)
    p_ver.add_argument("--output-dir", required=True)
    p_ver.add_argument("--num-shards", type=int, default=4)

    args = parser.parse_args()
    if args.cmd == "shard":
        run_shard(args)
    elif args.cmd == "verify-assembled":
        run_verify_assembled(args)


if __name__ == "__main__":
    main()

"""Resident multi-GPU frozen baseline constrained-beam service."""

from __future__ import annotations

from examples.experimental.generative_retriever.utils.paths import absolute_path

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
import gc
import logging
import multiprocessing as mp
from multiprocessing.connection import Connection
import os
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Annotated, Any

import anyio.to_thread
from fastapi import FastAPI, HTTPException
from filelock import FileLock
from pydantic import BaseModel, Field, field_validator
import torch
import uvicorn

from examples.experimental.generative_retriever.model.model_loading import load_policy_head_checkpoint
from examples.experimental.generative_retriever.model.transformers_infer import GenerationResult, SLMv3Retriever
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_TRAIN_GROUP_SIZE,
    MAX_EVAL_BATCH_SIZE,
)
from examples.experimental.generative_retriever.services.lru_cache import LRUCache

logger = logging.getLogger(__name__)
BASELINE_SERVICE_API_VERSION = 2
BASELINE_CACHE_SIZE = 8192
_SERVICE_API_VERSION = BASELINE_SERVICE_API_VERSION
_CACHE_SIZE = BASELINE_CACHE_SIZE


class BaselineRequest(BaseModel):
    queries: Annotated[list[str], Field(min_length=1, max_length=MAX_EVAL_BATCH_SIZE)]

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        if any(not query.strip() for query in queries):
            raise ValueError("Baseline queries must be non-empty")
        return queries


class LoadPolicyHeadRequest(BaseModel):
    policy_checkpoint: str


def _serialize_results(results: list[GenerationResult]) -> list[dict[str, Any]]:
    return [
        {
            "decoder_token_ids": result.decoder_token_ids,
            "positional_docid": result.positional_docid,
            "rq_codes": result.rq_codes,
            "token_logprobs": result.token_logprobs,
        }
        for result in results
    ]


def _baseline_worker_loop(
    conn: Connection,
    checkpoint: str,
    qwen_model: str,
    slm_root: str,
    id_mapping: str,
    device: str,
    policy_checkpoint: str | None,
) -> None:
    try:
        local_device = "cuda:0" if device.startswith("cuda") else device
        if local_device.startswith("cuda:"):
            torch.cuda.set_device(0)
        init_lock_path = Path(tempfile.gettempdir()) / "gr_baseline_init.lock"
        with FileLock(str(init_lock_path)):
            retriever = SLMv3Retriever(
                checkpoint=checkpoint,
                qwen_model=qwen_model,
                slm_root=slm_root,
                id_mapping=id_mapping,
                device=local_device,
            )
            if policy_checkpoint is not None:
                load_policy_head_checkpoint(retriever.model, policy_checkpoint)
                retriever.model.eval()
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 500, 500)
        conn.send(("ready", None))
        while True:
            msg = conn.recv()
            if msg is None:
                break
            cmd, payload = msg
            if cmd == "generate":
                queries, docids_per_query = payload
                generated = retriever.generate(
                    queries,
                    num_beams=docids_per_query,
                    num_return_sequences=docids_per_query,
                )
                conn.send(("ok", _serialize_results(generated)))
            elif cmd == "load_policy_head":
                tensor_count = load_policy_head_checkpoint(retriever.model, payload)
                retriever.model.eval()
                conn.send(("ok", tensor_count))
            else:
                conn.send(("error", f"Unknown command: {cmd}"))
    except BaseException as exc:
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        raise
    finally:
        conn.close()


class BaselinePool:
    def __init__(
        self,
        *,
        checkpoint: str,
        qwen_model: str,
        slm_root: str,
        id_mapping: str,
        devices: tuple[str, ...],
        cache_size: int,
        docids_per_query: int,
        policy_checkpoint: str | None = None,
    ) -> None:
        self.devices = devices
        self.worker_count = len(devices)
        self.cache: LRUCache[str, list[dict[str, Any]]] = LRUCache(cache_size)
        self.docids_per_query = docids_per_query
        self._cache_lock = Lock()
        ctx = mp.get_context("spawn")
        self._executor = ThreadPoolExecutor(max_workers=self.worker_count)
        self._conns: list[Connection] = []
        self._procs: list[Any] = []
        self._conn_locks = [Lock() for _ in range(self.worker_count)]
        saved_vis = {
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
        }
        try:
            for device in devices:
                if device.startswith("cuda:"):
                    gpu_id = device.split(":", 1)[1]
                    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
                    os.environ["HIP_VISIBLE_DEVICES"] = gpu_id
                    os.environ["NVIDIA_VISIBLE_DEVICES"] = gpu_id
                parent_conn, child_conn = ctx.Pipe(duplex=True)
                proc = ctx.Process(
                    target=_baseline_worker_loop,
                    args=(
                        child_conn,
                        checkpoint,
                        qwen_model,
                        slm_root,
                        id_mapping,
                        device,
                        policy_checkpoint,
                    ),
                    daemon=True,
                )
                proc.start()
                child_conn.close()
                self._conns.append(parent_conn)
                self._procs.append(proc)
        finally:
            for k, v in saved_vis.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        for idx, conn in enumerate(self._conns):
            status, detail = conn.recv()
            if status != "ready":
                self.close()
                raise RuntimeError(f"Baseline GPU subprocess {self.devices[idx]} failed to start: {detail}")

    def check_alive(self) -> None:
        for idx, proc in enumerate(self._procs):
            if not proc.is_alive():
                raise RuntimeError(f"Baseline GPU subprocess {self.devices[idx]} is not alive")

    def _call_worker(self, idx: int, cmd: str, payload: Any) -> Any:
        with self._conn_locks[idx]:
            proc = self._procs[idx]
            if not proc.is_alive():
                raise RuntimeError(f"Baseline GPU subprocess {self.devices[idx]} exited unexpectedly")
            conn = self._conns[idx]
            conn.send((cmd, payload))
            status, result = conn.recv()
        if status != "ok":
            raise RuntimeError(f"Baseline GPU subprocess {self.devices[idx]} error: {result}")
        return result

    def load_policy_head(self, policy_checkpoint: str) -> int:
        tensor_count = 0
        with self._cache_lock:
            for idx in range(self.worker_count):
                tensor_count = int(self._call_worker(idx, "load_policy_head", policy_checkpoint))
            self.cache = LRUCache(self.cache.capacity)
        return tensor_count

    def generate(self, request: BaselineRequest) -> dict[str, Any]:
        started = time.perf_counter()
        resolved: dict[int, list[dict[str, Any]]] = {}
        pending: dict[str, list[int]] = {}
        with self._cache_lock:
            for index, query in enumerate(request.queries):
                cached = self.cache.get(query)
                if cached is None:
                    pending.setdefault(query, []).append(index)
                else:
                    resolved[index] = cached

        buckets: list[list[str]] = [[] for _ in range(self.worker_count)]
        for pending_index, query in enumerate(pending):
            buckets[pending_index % self.worker_count].append(query)
        futures = []
        for idx, bucket in enumerate(buckets):
            if bucket:
                futures.append((
                    bucket,
                    self._executor.submit(
                        self._call_worker,
                        idx,
                        "generate",
                        (bucket, self.docids_per_query),
                    ),
                ))
        for bucket, future in futures:
            generated: list[dict[str, Any]] = future.result()
            if len(generated) != len(bucket) * self.docids_per_query:
                raise RuntimeError("Baseline worker returned an invalid beam group count")
            for query_index, query in enumerate(bucket):
                start = query_index * self.docids_per_query
                rows = generated[start : start + self.docids_per_query]
                for index in pending[query]:
                    resolved[index] = rows
                with self._cache_lock:
                    self.cache.put(query, rows)

        results = [
            {"query": query, "docids": resolved[index]}
            for index, query in enumerate(request.queries)
        ]
        return {
            "results": results,
            "docids_per_query": self.docids_per_query,
            "latency_seconds": time.perf_counter() - started,
            "cache": asdict(self.cache.stats()),
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        for conn in self._conns:
            try:
                conn.send(None)
                conn.close()
            except Exception:
                pass
        for proc in self._procs:
            if proc.is_alive():
                proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()


def _parse_devices(raw_devices: str) -> tuple[str, ...]:
    devices = tuple(device.strip() for device in raw_devices.split(",") if device.strip())
    return _validate_devices(devices)


def _validate_devices(devices: tuple[str, ...]) -> tuple[str, ...]:
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Baseline devices must be a non-empty tuple of distinct devices")
    return devices


def create_app(
    *,
    checkpoint: str,
    qwen_model: str,
    slm_root: str,
    id_mapping: str,
    docids_per_query: int = DEFAULT_TRAIN_GROUP_SIZE,
    devices: tuple[str, ...] = ("cuda:0", "cuda:1"),
    policy_checkpoint: str | None = None,
) -> FastAPI:
    devices = _validate_devices(devices)
    resolved_service_config = {
        "api_version": _SERVICE_API_VERSION,
        "checkpoint": str(absolute_path(checkpoint)),
        "qwen_model": str(absolute_path(qwen_model)),
        "slm_root": str(absolute_path(slm_root)),
        "id_mapping": str(absolute_path(id_mapping)),
        "devices": list(devices),
        "cache_size": _CACHE_SIZE,
        "docids_per_query": docids_per_query,
        **(
            {"policy_checkpoint": str(absolute_path(policy_checkpoint))}
            if policy_checkpoint is not None
            else {}
        ),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1024
        app.state.pool = BaselinePool(
            checkpoint=checkpoint,
            qwen_model=qwen_model,
            slm_root=slm_root,
            id_mapping=id_mapping,
            devices=devices,
            cache_size=_CACHE_SIZE,
            docids_per_query=docids_per_query,
            policy_checkpoint=policy_checkpoint,
        )
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 500, 500)
        try:
            yield
        finally:
            app.state.pool.close()
            del app.state.pool

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        pool: BaselinePool = app.state.pool
        try:
            pool.check_alive()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return {
            "status": "ready",
            "worker_count": pool.worker_count,
            "devices": pool.devices,
            "docids_per_query": pool.docids_per_query,
            "cache": asdict(pool.cache.stats()),
            "service_config": resolved_service_config,
        }

    @app.post("/generate")
    def generate(request: BaselineRequest) -> dict[str, Any]:
        try:
            return app.state.pool.generate(request)
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.post("/load_policy_head")
    def load_policy_head(request: LoadPolicyHeadRequest) -> dict[str, Any]:
        try:
            count = app.state.pool.load_policy_head(request.policy_checkpoint)
        except RuntimeError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        resolved_service_config["policy_checkpoint"] = str(
            absolute_path(request.policy_checkpoint)
        )
        return {"status": "ok", "tensors_loaded": count}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--slm-root", required=True)
    parser.add_argument("--id-mapping", required=True)
    parser.add_argument("--docids-per-query", type=int, default=DEFAULT_TRAIN_GROUP_SIZE)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30487)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(
            checkpoint=args.checkpoint,
            qwen_model=args.qwen_model,
            slm_root=args.slm_root,
            id_mapping=args.id_mapping,
            docids_per_query=args.docids_per_query,
            devices=_parse_devices(args.devices),
            policy_checkpoint=args.policy_checkpoint,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()

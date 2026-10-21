"""Serve keyword-averaged rewards using official relevance and prepared baselines.

Run:
    python -m examples.experimental.generative_retriever.reward_service \
        --baseline-url http://127.0.0.1:30487/generate \
        --mapping-url http://127.0.0.1:30488 \
        --mapping-version mapping.tsv --relevance-model /path/to/relevance-model \
        --port 30489
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictInt, create_model, field_validator

from examples.experimental.generative_retriever.id_text_mapping import _validate_id, lookup_service
from examples.experimental.generative_retriever.lru_cache import LRUCache
from examples.experimental.generative_retriever.relevance_model import (
    RelevanceScorer,
    load_relevance_pipeline_factory,
    resolve_relevance_source,
)
from examples.experimental.generative_retriever.relevance_reward import (
    RewardMode, build_baseline_set,
    calculate_keyword_rewards,
)

logger = logging.getLogger(__name__)
_SERVICE_API_VERSION = 9
_QUERY_CACHE_SIZE = 8192
_SCORE_CACHE_SIZE = 131072


class RewardSample(BaseModel):
    prompt: str
    docid_tokens: tuple[StrictInt, StrictInt, StrictInt, StrictInt]

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, query: str) -> str:
        if not query.strip():
            raise ValueError("Query must be non-empty")
        return query

    @field_validator("docid_tokens")
    @classmethod
    def validate_tokens(cls, tokens: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        _validate_id((tokens[0] - 2, tokens[1] - 2, tokens[2] - 2, tokens[3] - 2))
        return tokens


class RewardRequest(BaseModel):
    samples: Annotated[list[RewardSample], Field(min_length=1)]
    reward_mode: RewardMode = "discrete_relevance_unique"


def reward_request_model(max_samples: int) -> type[RewardRequest]:
    if max_samples <= 0:
        raise ValueError("Reward request sample limit must be positive")
    return create_model(
        f"RewardRequestMax{max_samples}",
        __base__=RewardRequest,
        samples=(list[RewardSample], Field(min_length=1, max_length=max_samples)),
    )


class BaselinePrepareRequest(BaseModel):
    queries: Annotated[list[str], Field(min_length=1, max_length=512)]

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        if any(not query.strip() for query in queries):
            raise ValueError("Baseline preparation queries must be non-empty")
        return queries


class KeywordRewardEngine:
    def __init__(
        self,
        baseline_url: str,
        mapping_url: str,
        mapping_version: str,
        relevance_model: str,
        device: str,
        batch_size: int,
        query_cache_size: int,
        score_cache_size: int,
        baseline_top_k: int,
        *,
        pair_batch_size: int,
        keyword_cache_size: int = 8192,
    ) -> None:
        self.mapping_version = mapping_version
        self.mapping_url = mapping_url
        self.baseline_url = baseline_url
        self.baseline_top_k = baseline_top_k
        self._baseline_queries: set[str] = set()
        self.baselines: LRUCache[str, frozenset[str]] = LRUCache(query_cache_size)
        relevance_source = resolve_relevance_source(relevance_model)
        pipeline_factory = load_relevance_pipeline_factory(relevance_source)
        pipeline = pipeline_factory(
            relevance_model, cup="large", device=device, dtype="bf16",
            batch_size=batch_size, cross_batch_size=256, show_progress=False,
        )
        self.scorer = RelevanceScorer(
            pipeline, feature_cache_size=query_cache_size, score_cache_size=score_cache_size,
            pair_batch_size=pair_batch_size, keyword_cache_size=keyword_cache_size,
        )
        self._lock = Lock()
        self._baseline_inflight: dict[str, Future[frozenset[str]]] = {}
        self._client = httpx.Client(timeout=300.0)

    def _cache_stats(self) -> dict[str, Any]:
        return {
            "baseline": asdict(self.baselines.stats()),
            "reward": asdict(self.scorer.scores.stats()),
            "query_features": asdict(self.scorer.query_features.stats()),
            "keyword_features": asdict(self.scorer.keyword_features.stats()),
        }

    def _lookup_docids(
        self,
        docids: list[tuple[int, int, int, int]],
    ) -> dict[tuple[int, int, int, int], tuple[str, ...]]:
        return lookup_service(
            self.mapping_url, docids,
            expected_mapping_version=self.mapping_version, client=self._client,
        )

    @staticmethod
    def _cache_delta(
        before: dict[str, dict[str, int]],
        after: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        return {
            name: {
                counter: after[name][counter] - before[name][counter]
                for counter in ("hits", "misses", "evictions")
            }
            for name in after
        }

    @staticmethod
    def _sum_cache_deltas(
        first: dict[str, dict[str, int]], second: dict[str, dict[str, int]],
    ) -> dict[str, dict[str, int]]:
        return {
            name: {counter: value + second[name][counter] for counter, value in counts.items()}
            for name, counts in first.items()
        }

    def _generate_baseline_docids(
        self, missing_queries: list[str],
    ) -> dict[str, list[tuple[int, int, int, int]]]:
        response = self._client.post(
            self.baseline_url,
            json={"queries": missing_queries},
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != len(missing_queries):
            raise ValueError("Baseline service returned an invalid query group")

        query_docids = {}
        for query, result in zip(missing_queries, results, strict=True):
            if not isinstance(result, dict) or result.get("query") != query:
                raise ValueError("Baseline service returned an invalid query group")
            rows = result.get("docids")
            if not isinstance(rows, list) or len(rows) != self.baseline_top_k:
                raise ValueError(
                    f"Baseline service must return exactly {self.baseline_top_k} doc IDs"
                )
            docids = []
            for row in rows:
                docid = row.get("positional_docid") if isinstance(row, dict) else None
                if (
                    not isinstance(docid, list)
                    or len(docid) != 4
                    or not all(isinstance(value, int) for value in docid)
                ):
                    raise ValueError("Baseline service returned an invalid positional doc ID")
                docids.append((docid[0], docid[1], docid[2], docid[3]))
            if len(set(docids)) != self.baseline_top_k:
                raise ValueError("Baseline service returned duplicate doc IDs")
            query_docids[query] = docids
        return query_docids

    def _prepare_baselines(
        self, queries: list[str],
    ) -> tuple[dict[str, frozenset[str]], dict[str, dict[str, int]]]:
        started = time.perf_counter()
        baselines: dict[str, frozenset[str]] = {}
        owned: dict[str, Future[frozenset[str]]] = {}
        waiting: dict[str, Future[frozenset[str]]] = {}
        with self._lock:
            before = self._cache_stats()
            for query in dict.fromkeys(queries):
                baseline = self.baselines.get(query)
                if baseline is not None:
                    baselines[query] = baseline
                elif query in self._baseline_inflight:
                    waiting[query] = self._baseline_inflight[query]
                else:
                    future: Future[frozenset[str]] = Future()
                    self._baseline_inflight[query] = future
                    owned[query] = future
            initial_delta = self._cache_delta(before, self._cache_stats())
        delta = initial_delta
        if owned:
            missing_queries = list(owned)
            try:
                if not self.baseline_url:
                    raise KeyError(f"Prepared baseline missing for queries {missing_queries!r}")
                query_docids = self._generate_baseline_docids(missing_queries)
                generated_at = time.perf_counter()
                mapping = self._lookup_docids([
                    docid for query in missing_queries for docid in query_docids[query]
                ])
                query_keywords = {
                    query: list(dict.fromkeys(
                        keyword
                        for docid in query_docids[query]
                        for keyword in mapping[docid]
                    ))
                    for query in missing_queries
                }
                mapped_at = time.perf_counter()
                with self._lock:
                    acquired_at = time.perf_counter()
                    before_scoring = self._cache_stats()
                    pair_scores = self.scorer.score([
                        (query, keyword)
                        for query in missing_queries
                        for keyword in query_keywords[query]
                    ])
                    scored_at = time.perf_counter()
                    for query in missing_queries:
                        scores = {
                            keyword: pair_scores[(query, keyword)]
                            for keyword in query_keywords[query]
                        }
                        baseline = build_baseline_set(scores)
                        self.baselines.put(query, baseline)
                        self._baseline_queries.add(query)
                        baselines[query] = baseline
                    delta = self._sum_cache_deltas(
                        initial_delta, self._cache_delta(before_scoring, self._cache_stats()),
                    )
                    for query, future in owned.items():
                        future.set_result(baselines[query])
                        del self._baseline_inflight[query]
                finished = time.perf_counter()
                logger.info(
                    "reward_baseline_stage queries=%d missing=%d baseline_s=%.3f mapping_s=%.3f "
                    "lock_wait_s=%.3f graphene_s=%.3f build_s=%.3f total_s=%.3f",
                    len(queries), len(owned), generated_at - started, mapped_at - generated_at,
                    acquired_at - mapped_at, scored_at - acquired_at, finished - scored_at,
                    finished - started,
                )
            except Exception as error:
                with self._lock:
                    for query, future in owned.items():
                        if not future.done():
                            future.set_exception(error)
                        self._baseline_inflight.pop(query, None)
                raise
        for query, future in waiting.items():
            baselines[query] = future.result()
        return baselines, delta

    def prepare_baselines(self, request: BaselinePrepareRequest) -> dict[str, Any]:
        started = time.perf_counter()
        baselines, delta = self._prepare_baselines(request.queries)
        with self._lock:
            cache = self._cache_stats()
        return {
            "prepared_queries": len(baselines),
            "mapping_version": self.mapping_version,
            "latency_seconds": time.perf_counter() - started,
            "cache": cache,
            "cache_delta": delta,
        }

    def score(self, request: RewardRequest) -> dict[str, Any]:
        started = time.perf_counter()
        docids = [
            (sample.docid_tokens[0] - 2, sample.docid_tokens[1] - 2,
             sample.docid_tokens[2] - 2, sample.docid_tokens[3] - 2)
            for sample in request.samples
        ]
        mapping = self._lookup_docids(docids)
        mapped_at = time.perf_counter()
        baselines, baseline_delta = self._prepare_baselines([
            sample.prompt for sample in request.samples
        ])
        prepared_at = time.perf_counter()
        with self._lock:
            acquired_at = time.perf_counter()
            before = self._cache_stats()
            sample_keywords = [
                list(dict.fromkeys(mapping[docid]))
                for docid in docids
            ]
            pairs = [
                (sample.prompt, keyword)
                for sample, keywords in zip(request.samples, sample_keywords, strict=True)
                for keyword in keywords
            ]
            pair_scores = self.scorer.score(pairs)
            scored_at = time.perf_counter()
            rewards = []
            details = []
            for sample, docid, keywords in zip(
                request.samples, docids, sample_keywords, strict=True,
            ):
                scores = {
                    keyword: pair_scores[(sample.prompt, keyword)]
                    for keyword in keywords
                }
                baseline = baselines[sample.prompt]
                breakdown = calculate_keyword_rewards(
                    scores, baseline, mode=request.reward_mode,
                )
                value = math.fsum(result.reward for result in breakdown.values()) / len(breakdown)
                rewards.append(value)
                details.append({
                    "query": sample.prompt, "docid": docid, "keyword_count": len(scores),
                    "aggregation": "mean", "reward": value,
                    "keywords": [{"keyword": keyword, **asdict(result)}
                                 for keyword, result in breakdown.items()],
                })
                logger.debug("reward_mode=%s docid=%s keywords=%d reward=%f",
                            request.reward_mode, docid, len(scores), value)
            cache = self._cache_stats()
        finished = time.perf_counter()
        logger.info(
            "reward_score_stage samples=%d pairs=%d mapping_s=%.3f lock_wait_s=%.3f "
            "baseline_s=%.3f graphene_s=%.3f breakdown_s=%.3f total_s=%.3f",
            len(request.samples), len(pairs), mapped_at - started, acquired_at - prepared_at,
            prepared_at - mapped_at, scored_at - acquired_at, finished - scored_at,
            finished - started,
        )
        return {
            "rewards": rewards,
            "details": details,
            "reward_mode": request.reward_mode,
            "mapping_version": self.mapping_version,
            "latency_seconds": finished - started,
            "cache": cache,
            "cache_delta": self._sum_cache_deltas(
                baseline_delta, self._cache_delta(before, cache),
            ),
        }

    def close(self) -> None:
        self._client.close()


def create_app(
    baseline_url: str,
    mapping_url: str,
    mapping_version: str,
    relevance_model: str,
    *,
    batch_size: int = 2048,
    baseline_top_k: int = 64,
    pair_batch_size: int = 8192,
    max_samples_per_request: int = 2048,
    keyword_cache_size: int = _QUERY_CACHE_SIZE,
    score_cache_size: int = _SCORE_CACHE_SIZE,
) -> FastAPI:
    if batch_size < 4 or batch_size % 4:
        raise ValueError("batch_size must be a positive multiple of 4")
    if keyword_cache_size <= 0 or score_cache_size <= 0:
        raise ValueError("Reward cache capacities must be positive")
    request_model = reward_request_model(max_samples_per_request)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = KeywordRewardEngine(
            baseline_url,
            mapping_url,
            mapping_version,
            relevance_model,
            "cuda:0",
            batch_size,
            _QUERY_CACHE_SIZE,
            score_cache_size,
            baseline_top_k,
            pair_batch_size=pair_batch_size,
            keyword_cache_size=keyword_cache_size,
        )
        try:
            yield
        finally:
            app.state.engine.close()
            del app.state.engine

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        engine = app.state.engine
        return {
            "status": "ready",
            "mode": "worker",
            "worker_count": 1,
            "device": "cuda:0",
            "mapping_version": engine.mapping_version,
            "baseline_queries": len(engine._baseline_queries),
            "cache": engine._cache_stats(),
            "service_config": {
                "api_version": _SERVICE_API_VERSION,
                "baseline_url": baseline_url,
                "mapping_url": mapping_url,
                "mapping_version": mapping_version,
                "relevance_source": str(resolve_relevance_source(relevance_model)),
                "relevance_model": str(Path(relevance_model).expanduser().resolve()),
                "query_cache_size": _QUERY_CACHE_SIZE,
                "keyword_cache_size": keyword_cache_size,
                "score_cache_size": score_cache_size,
                "batch_size": batch_size,
                "pair_batch_size": engine.scorer.pair_batch_size,
                "baseline_top_k": baseline_top_k,
                "max_samples_per_request": max_samples_per_request,
            },
        }

    def score(request: RewardRequest) -> dict[str, Any]:
        try:
            return app.state.engine.score(request)
        except KeyError as error:
            logger.error("Baseline lookup failed: %s", error)
            raise HTTPException(status_code=409, detail=str(error)) from error
        except LookupError as error:
            logger.error("Official mapping lookup failed: %s", error)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except httpx.HTTPError as error:
            logger.error("Dependent service unavailable: %s", error)
            raise HTTPException(status_code=502, detail=str(error)) from error

    score.__annotations__["request"] = request_model
    app.post("/score")(score)

    @app.post("/prepare")
    def prepare(request: BaselinePrepareRequest) -> dict[str, Any]:
        try:
            return app.state.engine.prepare_baselines(request)
        except KeyError as error:
            logger.error("Baseline preparation failed: %s", error)
            raise HTTPException(status_code=409, detail=str(error)) from error
        except LookupError as error:
            logger.error("Official mapping lookup failed during preparation: %s", error)
            raise HTTPException(status_code=422, detail=str(error)) from error
        except httpx.HTTPError as error:
            logger.error("Dependent service unavailable during preparation: %s", error)
            raise HTTPException(status_code=502, detail=str(error)) from error
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--mapping-url", required=True)
    parser.add_argument("--mapping-version", required=True)
    parser.add_argument("--relevance-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30489)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--baseline-top-k", type=int, default=64)
    parser.add_argument("--max-samples-per-request", type=int, default=2048)
    parser.add_argument("--keyword-cache-size", type=int, default=_QUERY_CACHE_SIZE)
    parser.add_argument("--score-cache-size", type=int, default=_SCORE_CACHE_SIZE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(
            args.baseline_url,
            args.mapping_url,
            mapping_version=args.mapping_version,
            relevance_model=args.relevance_model,
            batch_size=args.batch_size,
            baseline_top_k=args.baseline_top_k,
            pair_batch_size=args.pair_batch_size,
            max_samples_per_request=args.max_samples_per_request,
            keyword_cache_size=args.keyword_cache_size,
            score_cache_size=args.score_cache_size,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()

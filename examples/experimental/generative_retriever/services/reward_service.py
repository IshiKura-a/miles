"""Serve keyword-averaged rewards using official relevance and prepared baselines.

Run:
    python -m examples.experimental.generative_retriever.services.reward_service \
        --baseline-url http://127.0.0.1:30487/generate \
        --mapping-url http://127.0.0.1:30488 \
        --mapping-version mapping.tsv --relevance-model /path/to/relevance-model \
        --port 30489
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import time
from collections.abc import Sequence
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

from examples.experimental.generative_retriever.index.id_text_mapping import lookup_service, validate_docid
from examples.experimental.generative_retriever.services.lru_cache import LRUCache
from examples.experimental.generative_retriever.reward.relevance_model import (
    DEFAULT_KEYWORD_CACHE_SIZE,
    DEFAULT_QUERY_CACHE_SIZE,
    DEFAULT_SCORE_CACHE_SIZE,
    MultiGpuRelevanceScorer,
    resolve_relevance_source,
)
from examples.experimental.generative_retriever.reward.relevance_reward import (
    DEFAULT_REWARD_BATCH_SIZE,
    DEFAULT_TRAIN_GROUP_SIZE,
    DOC2K_HEAD_RANK_CUTOFF,
    DOC2K_MAX_ADDS_PER_QUERY,
    DOC2K_MIN_GROUP_SAMPLES,
    DOC2K_MIN_HEAD_RANKS,
    DOC2K_PRUNE_MIN_QUERIES,
    DOCID_REWARD_FLOOR,
    DOCID_TOKEN_OFFSET,
    MAX_DOCIDS_PER_QUERY,
    REWARD_PAIR_BATCH_SIZE,
    REWARD_TOWER_BATCH_SIZE,
    RQ_CODE_NUM,
    BaselineSets,
    RewardMode,
    build_baseline_set,
    calculate_keyword_rewards,
)

logger = logging.getLogger(__name__)
REWARD_SERVICE_API_VERSION = 12
_SERVICE_API_VERSION = REWARD_SERVICE_API_VERSION
_QUERY_CACHE_SIZE = DEFAULT_QUERY_CACHE_SIZE
_KEYWORD_CACHE_SIZE = DEFAULT_KEYWORD_CACHE_SIZE
_SCORE_CACHE_SIZE = DEFAULT_SCORE_CACHE_SIZE
_CROSS_BATCH_SIZE = 8192
_INLINE_KEYWORDS_SAMPLE_LIMIT = 16


def _parse_devices(raw_devices: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw_devices, str):
        devices = tuple(d.strip() for d in raw_devices.split(",") if d.strip())
    else:
        devices = tuple(str(d).strip() for d in raw_devices if str(d).strip())
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Reward devices must be a non-empty tuple of distinct devices")
    return devices


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
        validate_docid((
            tokens[0] - DOCID_TOKEN_OFFSET,
            tokens[1] - DOCID_TOKEN_OFFSET,
            tokens[2] - DOCID_TOKEN_OFFSET,
            tokens[3] - DOCID_TOKEN_OFFSET,
        ))
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
    queries: Annotated[list[str], Field(min_length=1, max_length=MAX_DOCIDS_PER_QUERY)]

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        if any(not query.strip() for query in queries):
            raise ValueError("Baseline preparation queries must be non-empty")
        return queries


def summarize_breakdown_metrics(breakdown: dict[str, Any]) -> dict[str, float]:
    items = list(breakdown.values())
    n = float(len(items))
    metrics = {
        "pm_score": math.fsum(r.pm_score for r in items) / n,
        "sm_score": math.fsum(r.sm_score for r in items) / n,
        "pm_pass": math.fsum(r.pm_pass for r in items) / n,
        "sm_pass": math.fsum(r.sm_pass for r in items) / n,
        "normalized_pm": math.fsum(r.normalized_pm for r in items) / n,
        "normalized_sm": math.fsum(r.normalized_sm for r in items) / n,
        "pm_relevance": math.fsum(r.pm_relevance for r in items) / n,
        "sm_relevance": math.fsum(r.sm_relevance for r in items) / n,
        "relevance_quality": math.fsum(r.relevance_quality for r in items) / n,
        "pm_unique_bonus": math.fsum(r.pm_unique_bonus for r in items) / n,
        "sm_unique_bonus": math.fsum(r.sm_unique_bonus for r in items) / n,
        "unique_bonus": math.fsum(r.unique_bonus for r in items) / n,
        "discrete_base": math.fsum(r.discrete_base for r in items) / n,
        "continuous_base": math.fsum(r.continuous_base for r in items) / n,
        "discrete_total": math.fsum(r.discrete_total for r in items) / n,
        "continuous_total": math.fsum(r.continuous_total for r in items) / n,
        "reward": math.fsum(r.reward for r in items) / n,
        "pm_unique_keyword_rate": math.fsum(float(r.is_pm_unique) for r in items) / n,
        "sm_unique_keyword_rate": math.fsum(float(r.is_sm_unique) for r in items) / n,
        "unique_keyword_rate": math.fsum(float(r.is_unique) for r in items) / n,
        "relevance_eligible_rate": math.fsum(float(r.relevance_eligible) for r in items) / n,
        "pm_unique_eligible_rate": math.fsum(float(r.pm_unique_eligible) for r in items) / n,
        "sm_unique_eligible_rate": math.fsum(float(r.sm_unique_eligible) for r in items) / n,
        "unique_eligible_rate": math.fsum(float(r.unique_eligible) for r in items) / n,
    }
    unique_qual = [r.relevance_quality for r in items if r.is_pm_unique or r.is_sm_unique]
    metrics["unique_quality"] = math.fsum(unique_qual) / len(unique_qual) if unique_qual else 0.0
    metrics["train_total"] = metrics["reward"]
    return metrics


class KeywordRewardEngine:
    def __init__(
        self,
        baseline_url: str,
        mapping_url: str,
        mapping_version: str,
        relevance_model: str,
        devices: str | Sequence[str],
        batch_size: int,
        query_cache_size: int,
        score_cache_size: int,
        docids_per_query: int,
        *,
        pair_batch_size: int,
        keyword_cache_size: int = _KEYWORD_CACHE_SIZE,
    ) -> None:
        self.devices = _parse_devices(devices)
        self.mapping_version = mapping_version
        self.mapping_url = mapping_url
        self.baseline_url = baseline_url
        self.docids_per_query = docids_per_query
        self._baseline_queries: set[str] = set()
        self.baselines: LRUCache[str, BaselineSets] = LRUCache(query_cache_size)
        self.scorer = MultiGpuRelevanceScorer(
            relevance_model,
            self.devices,
            batch_size=batch_size,
            cross_batch_size=_CROSS_BATCH_SIZE,
            feature_cache_size=query_cache_size,
            score_cache_size=score_cache_size,
            pair_batch_size=pair_batch_size,
            keyword_cache_size=keyword_cache_size,
        )
        self._lock = Lock()
        self._baseline_inflight: dict[str, Future[BaselineSets]] = {}
        self._client = httpx.Client(
            timeout=300.0,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=0),
            trust_env=False,
        )
        self._step_additions: dict[tuple[int, int, int, int], list[str]] = {}
        self._step_docid_queries: dict[tuple[int, int, int, int], set[str]] = {}
        self._step_docid_keywords: dict[tuple[int, int, int, int], list[str]] = {}
        self._step_edge_passes: dict[tuple[tuple[int, int, int, int], str], int] = {}

    def _score_pairs_local(
        self,
        pairs: list[tuple[str, str]],
    ) -> tuple[dict[tuple[str, str], tuple[float, float]], dict[str, dict[str, int]]]:
        return self.scorer.score_with_deltas(pairs)

    def _cache_stats(self) -> dict[str, Any]:
        return {
            "baseline": asdict(self.baselines.stats()),
            "reward": asdict(self.scorer.scores.stats()),
            "query_features": asdict(self.scorer.query_feature_stats()),
            "keyword_features": asdict(self.scorer.keyword_feature_stats()),
        }

    def _lookup_docids(
        self,
        docids: list[tuple[int, int, int, int]],
        *,
        use_initial: bool = False,
    ) -> dict[tuple[int, int, int, int], tuple[str, ...]]:
        return lookup_service(
            self.mapping_url, docids,
            expected_mapping_version=self.mapping_version,
            use_initial=use_initial,
            client=self._client,
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
            if not isinstance(rows, list) or len(rows) != self.docids_per_query:
                raise ValueError(
                    f"Baseline service must return exactly {self.docids_per_query} doc IDs"
                )
            docids = []
            for row in rows:
                docid = row.get("positional_docid") if isinstance(row, dict) else None
                if (
                    not isinstance(docid, list)
                    or len(docid) != RQ_CODE_NUM
                    or not all(isinstance(value, int) for value in docid)
                ):
                    raise ValueError("Baseline service returned an invalid positional doc ID")
                docids.append((docid[0], docid[1], docid[2], docid[3]))
            if len(set(docids)) != self.docids_per_query:
                raise ValueError("Baseline service returned duplicate doc IDs")
            query_docids[query] = docids
        return query_docids

    def _prepare_baselines(
        self, queries: list[str],
    ) -> tuple[dict[str, BaselineSets], dict[str, dict[str, int]]]:
        started = time.perf_counter()
        baselines: dict[str, BaselineSets] = {}
        owned: dict[str, Future[BaselineSets]] = {}
        waiting: dict[str, Future[BaselineSets]] = {}
        with self._lock:
            before = self._cache_stats()
            for query in dict.fromkeys(queries):
                if query in self._baseline_inflight:
                    waiting[query] = self._baseline_inflight[query]
                    continue
                baseline = self.baselines.get(query)
                if baseline is not None:
                    baselines[query] = baseline
                else:
                    future: Future[BaselineSets] = Future()
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
                mapping = self._lookup_docids(
                    [
                        docid for query in missing_queries for docid in query_docids[query]
                    ],
                    use_initial=True,
                )
                query_keywords = {
                    query: list(dict.fromkeys(
                        keyword
                        for docid in query_docids[query]
                        for keyword in mapping[docid]
                    ))
                    for query in missing_queries
                }
                mapped_at = time.perf_counter()
                pair_scores, scoring_delta = self._score_pairs_local([
                    (query, keyword)
                    for query in missing_queries
                    for keyword in query_keywords[query]
                ])
                scored_at = time.perf_counter()
                with self._lock:
                    acquired_at = time.perf_counter()
                    b_evictions_before = self.baselines.stats().evictions
                    for query in missing_queries:
                        scores = {
                            keyword: pair_scores[(query, keyword)]
                            for keyword in query_keywords[query]
                        }
                        baseline = build_baseline_set(scores)
                        self.baselines.put(query, baseline)
                        self._baseline_queries.add(query)
                        baselines[query] = baseline
                    b_evictions = self.baselines.stats().evictions - b_evictions_before
                    scoring_delta["baseline"]["evictions"] += b_evictions
                    delta = self._sum_cache_deltas(initial_delta, scoring_delta)
                    for query, future in owned.items():
                        future.set_result(baselines[query])
                        del self._baseline_inflight[query]
                finished = time.perf_counter()
                logger.info(
                    "reward_baseline_stage queries=%d missing=%d baseline_s=%.3f mapping_s=%.3f "
                    "lock_wait_s=%.3f graphene_s=%.3f build_s=%.3f total_s=%.3f",
                    len(queries), len(owned), generated_at - started, mapped_at - generated_at,
                    acquired_at - scored_at, scored_at - mapped_at, finished - acquired_at,
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
            (
                sample.docid_tokens[0] - DOCID_TOKEN_OFFSET,
                sample.docid_tokens[1] - DOCID_TOKEN_OFFSET,
                sample.docid_tokens[2] - DOCID_TOKEN_OFFSET,
                sample.docid_tokens[3] - DOCID_TOKEN_OFFSET,
            )
            for sample in request.samples
        ]
        mapping = self._lookup_docids(docids)
        mapped_at = time.perf_counter()
        baselines, baseline_delta = self._prepare_baselines([
            sample.prompt for sample in request.samples
        ])
        prepared_at = time.perf_counter()
        sample_keywords = [
            list(dict.fromkeys(mapping[docid]))
            for docid in docids
        ]
        pairs = [
            (sample.prompt, keyword)
            for sample, keywords in zip(request.samples, sample_keywords, strict=True)
            for keyword in keywords
        ]
        pair_scores, scoring_delta = self._score_pairs_local(pairs)
        scored_at = time.perf_counter()
        query_kw_freq: dict[str, dict[str, int]] = {}
        query_group_indices: dict[str, list[int]] = {}
        for idx, (sample, keywords) in enumerate(zip(request.samples, sample_keywords, strict=True)):
            freq_map = query_kw_freq.setdefault(sample.prompt, {})
            for keyword in keywords:
                freq_map[keyword] = freq_map.get(keyword, 0) + 1
            query_group_indices.setdefault(sample.prompt, []).append(idx)
        rewards = []
        details = []
        sample_breakdowns: list[dict[str, Any]] = []
        for sample, docid, keywords in zip(
            request.samples, docids, sample_keywords, strict=True,
        ):
            scores = {
                keyword: pair_scores[(sample.prompt, keyword)]
                for keyword in keywords
            }
            baseline = baselines[sample.prompt]
            breakdown = calculate_keyword_rewards(
                scores,
                baseline,
                mode=request.reward_mode,
                keyword_frequencies=query_kw_freq.get(sample.prompt),
            )
            sample_breakdowns.append(breakdown)
            denominator = max(len(breakdown), DOCID_REWARD_FLOOR)
            value = math.fsum(result.reward for result in breakdown.values()) / float(denominator)
            rewards.append(value)
            detail_entry: dict[str, Any] = {
                "query": sample.prompt, "docid": docid, "keyword_count": len(scores),
                "aggregation": f"floor_{DOCID_REWARD_FLOOR}", "reward": value,
                "metrics": summarize_breakdown_metrics(breakdown),
            }
            if len(request.samples) <= _INLINE_KEYWORDS_SAMPLE_LIMIT:
                detail_entry["keywords"] = [
                    {"keyword": keyword, **asdict(result)}
                    for keyword, result in breakdown.items()
                ]
            details.append(detail_entry)
            logger.debug("reward_mode=%s docid=%s keywords=%d reward=%f",
                        request.reward_mode, docid, len(scores), value)
        with self._lock:
            acquired_at = time.perf_counter()
            for sample, docid, keywords, breakdown in zip(
                request.samples, docids, sample_keywords, sample_breakdowns, strict=True
            ):
                if len(keywords) > DOCID_REWARD_FLOOR:
                    self._step_docid_queries.setdefault(docid, set()).add(sample.prompt)
                    self._step_docid_keywords.setdefault(docid, keywords)
                    for keyword, result in breakdown.items():
                        edge = (docid, keyword)
                        self._step_edge_passes[edge] = (
                            self._step_edge_passes.get(edge, 0) + int(result.relevance_eligible)
                        )
            for indices in query_group_indices.values():
                if len(indices) < DOC2K_MIN_GROUP_SAMPLES:
                    continue
                head_cutoff = min(DOC2K_HEAD_RANK_CUTOFF, max(DOC2K_MIN_HEAD_RANKS, len(indices) // 2))
                head_indices = indices[:head_cutoff]
                tail_indices = indices[head_cutoff:]
                if not tail_indices:
                    continue
                head_keywords = {
                    kw for i in head_indices for kw in sample_keywords[i]
                }
                tail_unique_good: list[str] = []
                seen_tail: set[str] = set()
                for i in tail_indices:
                    for kw, res in sample_breakdowns[i].items():
                        if res.unique_eligible and kw not in head_keywords and kw not in seen_tail:
                            tail_unique_good.append(kw)
                            seen_tail.add(kw)
                if not tail_unique_good:
                    continue
                target_docid = docids[head_indices[0]]
                for i in head_indices:
                    if len(sample_keywords[i]) < DOCID_REWARD_FLOOR:
                        target_docid = docids[i]
                        break
                bucket_adds = self._step_additions.setdefault(target_docid, [])
                existing_adds = set(bucket_adds)
                for kw in tail_unique_good[:DOC2K_MAX_ADDS_PER_QUERY]:
                    if kw not in existing_adds:
                        bucket_adds.append(kw)
                        existing_adds.add(kw)
            cache = self._cache_stats()
        finished = time.perf_counter()
        logger.info(
            "reward_score_stage samples=%d pairs=%d mapping_s=%.3f lock_wait_s=%.3f "
            "baseline_s=%.3f graphene_s=%.3f breakdown_s=%.3f total_s=%.3f",
            len(request.samples), len(pairs), mapped_at - started, acquired_at - scored_at,
            prepared_at - mapped_at, scored_at - prepared_at, finished - scored_at,
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
                baseline_delta, scoring_delta,
            ),
        }

    def harvest_step_delta(self, min_queries: int = DOC2K_PRUNE_MIN_QUERIES) -> dict[str, Any]:
        with self._lock:
            additions = [
                {"docid": list(docid), "keywords": list(keywords)}
                for docid, keywords in self._step_additions.items()
                if keywords
            ]
            removals = []
            for docid, queries in self._step_docid_queries.items():
                if len(queries) < min_queries:
                    continue
                keywords = self._step_docid_keywords.get(docid, [])
                if len(keywords) <= DOCID_REWARD_FLOOR:
                    continue
                dead_keywords = [
                    kw for kw in keywords
                    if self._step_edge_passes.get((docid, kw), 0) == 0
                ]
                max_removable = max(0, len(keywords) - DOCID_REWARD_FLOOR)
                if dead_keywords and max_removable > 0:
                    removals.append({
                        "docid": list(docid),
                        "keywords": dead_keywords[:max_removable],
                    })
            self._step_additions.clear()
            self._step_docid_queries.clear()
            self._step_docid_keywords.clear()
            self._step_edge_passes.clear()
            return {
                "additions": additions,
                "removals": removals,
                "mapping_url": self.mapping_url,
            }

    def close(self) -> None:
        self.scorer.close()
        self._client.close()


def create_app(
    baseline_url: str,
    mapping_url: str,
    mapping_version: str,
    relevance_model: str,
    *,
    devices: str | Sequence[str] = ("cuda:0",),
    batch_size: int = REWARD_TOWER_BATCH_SIZE,
    docids_per_query: int = DEFAULT_TRAIN_GROUP_SIZE,
    pair_batch_size: int = REWARD_PAIR_BATCH_SIZE,
    max_samples_per_request: int = DEFAULT_REWARD_BATCH_SIZE,
    query_cache_size: int = _QUERY_CACHE_SIZE,
    keyword_cache_size: int = _KEYWORD_CACHE_SIZE,
    score_cache_size: int = _SCORE_CACHE_SIZE,
) -> FastAPI:
    parsed_devices = _parse_devices(devices)
    if batch_size < 4 or batch_size % 4:
        raise ValueError("batch_size must be a positive multiple of 4")
    if query_cache_size <= 0 or keyword_cache_size <= 0 or score_cache_size <= 0:
        raise ValueError("Reward cache capacities must be positive")
    request_model = reward_request_model(max_samples_per_request)
    resolved_relevance_source = str(resolve_relevance_source(relevance_model))
    resolved_relevance_model = str(Path(relevance_model).expanduser().resolve())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = KeywordRewardEngine(
            baseline_url,
            mapping_url,
            mapping_version,
            relevance_model,
            parsed_devices,
            batch_size,
            query_cache_size,
            score_cache_size,
            docids_per_query,
            pair_batch_size=pair_batch_size,
            keyword_cache_size=keyword_cache_size,
        )
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 500, 500)
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
            "worker_count": len(parsed_devices),
            "devices": list(parsed_devices),
            "mapping_version": engine.mapping_version,
            "baseline_queries": len(engine._baseline_queries),
            "cache": engine._cache_stats(),
            "service_config": {
                "api_version": _SERVICE_API_VERSION,
                "baseline_url": baseline_url,
                "mapping_url": mapping_url,
                "mapping_version": mapping_version,
                "relevance_source": resolved_relevance_source,
                "relevance_model": resolved_relevance_model,
                "devices": list(parsed_devices),
                "query_cache_size": query_cache_size,
                "keyword_cache_size": keyword_cache_size,
                "score_cache_size": score_cache_size,
                "batch_size": batch_size,
                "pair_batch_size": engine.scorer.pair_batch_size,
                "docids_per_query": docids_per_query,
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

    @app.post("/harvest_step_delta")
    def harvest_step_delta() -> dict[str, Any]:
        return app.state.engine.harvest_step_delta()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--mapping-url", required=True)
    parser.add_argument("--mapping-version", required=True)
    parser.add_argument("--relevance-model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30489)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=REWARD_TOWER_BATCH_SIZE)
    parser.add_argument("--pair-batch-size", type=int, default=REWARD_PAIR_BATCH_SIZE)
    parser.add_argument("--docids-per-query", type=int, default=DEFAULT_TRAIN_GROUP_SIZE)
    parser.add_argument("--max-samples-per-request", type=int, default=DEFAULT_REWARD_BATCH_SIZE)
    parser.add_argument("--query-cache-size", type=int, default=_QUERY_CACHE_SIZE)
    parser.add_argument("--keyword-cache-size", type=int, default=_KEYWORD_CACHE_SIZE)
    parser.add_argument("--score-cache-size", type=int, default=_SCORE_CACHE_SIZE)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(
            args.baseline_url,
            args.mapping_url,
            mapping_version=args.mapping_version,
            relevance_model=args.relevance_model,
            devices=_parse_devices(args.devices),
            batch_size=args.batch_size,
            docids_per_query=args.docids_per_query,
            pair_batch_size=args.pair_batch_size,
            max_samples_per_request=args.max_samples_per_request,
            query_cache_size=args.query_cache_size,
            keyword_cache_size=args.keyword_cache_size,
            score_cache_size=args.score_cache_size,
        ),
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()

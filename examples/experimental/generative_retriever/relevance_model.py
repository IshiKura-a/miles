"""Resolve the source snapshot shipped with a Graphene relevance checkpoint."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from examples.experimental.generative_retriever.lru_cache import LRUCache

EXPECTED_GRAPHENE_COMMIT = "a57cb597de0891c1e0a2612d12366188b96aa0cc"
_PIPELINE_PATH = Path("graphene/multi_cup/multi_cup_graphene_pipeline.py")


class RelevancePipeline(Protocol):
    def encode_query(self, queries: list[str]) -> np.ndarray: ...

    def encode_keyword(self, keywords: list[str]) -> np.ndarray: ...

    def cross_score(self, query_vectors: np.ndarray, keyword_vectors: np.ndarray) -> np.ndarray: ...


class RelevancePipelineFactory(Protocol):
    def __call__(
        self,
        model_path: str,
        cup: str,
        device: str,
        dtype: str,
        batch_size: int,
        cross_batch_size: int,
        show_progress: bool,
    ) -> RelevancePipeline: ...


class RelevanceScorer:
    """Batch distinct tower inputs; keep active results independent of LRU eviction.

    The caller serializes access to the pipeline and caches.
    """

    def __init__(
        self,
        pipeline: RelevancePipeline,
        *,
        feature_cache_size: int = 8192,
        score_cache_size: int = 131072,
        pair_batch_size: int = 4096,
    ) -> None:
        if pair_batch_size <= 0:
            raise ValueError("Relevance pair batch size must be positive")
        self.pipeline = pipeline
        self.pair_batch_size = pair_batch_size
        self.query_features: LRUCache[str, np.ndarray] = LRUCache(feature_cache_size)
        self.keyword_features: LRUCache[str, np.ndarray] = LRUCache(feature_cache_size)
        self.scores: LRUCache[tuple[str, str], tuple[float, float]] = LRUCache(
            score_cache_size
        )

    def _encode(
        self, texts: list[str], *, query: bool,
    ) -> dict[str, np.ndarray]:
        cache = self.query_features if query else self.keyword_features
        resolved = {}
        missing = []
        for text in dict.fromkeys(texts):
            features = cache.get(text)
            if features is None:
                missing.append(text)
            else:
                resolved[text] = features
        if missing:
            encode = self.pipeline.encode_query if query else self.pipeline.encode_keyword
            encoded = encode(missing)
            if (
                encoded.ndim != 3
                or encoded.shape[0] != len(missing)
                or not np.isfinite(encoded).all()
            ):
                raise RuntimeError("Official relevance model returned invalid tower features")
            for index, text in enumerate(missing):
                # Copy so a cached row does not retain the entire encoded batch.
                features = encoded[index:index + 1].copy()
                cache.put(text, features)
                resolved[text] = features
        return resolved

    def score(self, pairs: list[tuple[str, str]]) -> dict[tuple[str, str], tuple[float, float]]:
        resolved = {}
        missing = []
        for pair in dict.fromkeys(pairs):
            score = self.scores.get(pair)
            if score is None:
                missing.append(pair)
            else:
                resolved[pair] = score
        for start in range(0, len(missing), self.pair_batch_size):
            chunk = missing[start:start + self.pair_batch_size]
            queries = self._encode([query for query, _ in chunk], query=True)
            keywords = self._encode([keyword for _, keyword in chunk], query=False)
            scores = self.pipeline.cross_score(
                np.concatenate([queries[query] for query, _ in chunk], axis=0),
                np.concatenate([keywords[keyword] for _, keyword in chunk], axis=0),
            )
            if (
                scores.shape != (len(chunk), 4)
                or not np.isfinite(scores).all()
                or not ((scores >= 0) & (scores <= 1)).all()
            ):
                raise RuntimeError("Official relevance model returned invalid scores")
            for pair, row in zip(chunk, scores, strict=True):
                score = (float(row[1]), float(row[2]))
                self.scores.put(pair, score)
                resolved[pair] = score
        return resolved


def _validate_source(path: Path) -> Path:
    source = path.expanduser().resolve()
    if not (source / _PIPELINE_PATH).is_file():
        raise FileNotFoundError(f"Graphene pipeline is missing from relevance source: {source}")
    commit_path = source / "GIT_COMMIT.txt"
    commit = commit_path.read_text(encoding="utf-8").strip()
    if commit != EXPECTED_GRAPHENE_COMMIT:
        raise ValueError(
            f"Graphene source commit mismatch: expected {EXPECTED_GRAPHENE_COMMIT}, got {commit!r}"
        )
    return source


def resolve_relevance_source(
    relevance_model: str | Path,
    relevance_source: str | Path | None = None,
) -> Path:
    if relevance_source is not None:
        return _validate_source(Path(relevance_source))

    model = Path(relevance_model).expanduser().resolve()
    pointer = model / "SOURCE_CHECKPOINT.txt"
    if not pointer.is_file():
        raise FileNotFoundError(f"Relevance model has no SOURCE_CHECKPOINT.txt: {model}")
    source_checkpoint = Path(pointer.read_text(encoding="utf-8").strip())
    direct_source = source_checkpoint / "code"
    if direct_source.is_dir():
        return _validate_source(direct_source)

    cosmos_root = os.environ.get("cosmos_data_path")
    marker = "/local/"
    source_text = os.fspath(source_checkpoint)
    if not cosmos_root or marker not in source_text:
        raise FileNotFoundError(
            "Cannot resolve relevance source snapshot; set cosmos_data_path or pass --relevance-source"
        )
    relative_checkpoint = source_text.split(marker, 1)[1]
    return _validate_source(Path(cosmos_root) / "local" / relative_checkpoint / "code")


def load_relevance_pipeline_factory(source: str | Path) -> RelevancePipelineFactory:
    resolved = _validate_source(Path(source))
    source_text = os.fspath(resolved)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("graphene.multi_cup.multi_cup_graphene_pipeline")
    factory = getattr(module, "MultiCupGraphenePipeline", None)
    if not callable(factory):
        raise TypeError("Graphene source does not expose MultiCupGraphenePipeline")
    return cast(RelevancePipelineFactory, factory)

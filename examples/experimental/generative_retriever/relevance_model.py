"""Resolve the source snapshot shipped with a Graphene relevance checkpoint."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Protocol, cast

EXPECTED_GRAPHENE_COMMIT = "a57cb597de0891c1e0a2612d12366188b96aa0cc"
_PIPELINE_PATH = Path("graphene/multi_cup/multi_cup_graphene_pipeline.py")


class RelevancePipeline(Protocol):
    def encode_query(self, queries: list[str]) -> Any: ...

    def encode_keyword(self, keywords: list[str]) -> Any: ...

    def cross_score(self, query_vectors: Any, keyword_vectors: Any) -> Any: ...


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

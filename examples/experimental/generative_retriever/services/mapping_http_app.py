"""FastAPI HTTP service wrapper and wire schemas around ResidentMapping."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, StrictInt, field_validator
import uvicorn

from examples.experimental.generative_retriever.index.id_text_mapping import (
    PositionalID,
    validate_docid,
)
from examples.experimental.generative_retriever.reward.relevance_reward import DOCID_MAX_KEYWORDS
from examples.experimental.generative_retriever.utils.paths import absolute_path

if TYPE_CHECKING:
    from examples.experimental.generative_retriever.services.mapping_service import ResidentMapping


class LookupRequest(BaseModel):
    docids: Annotated[list[tuple[StrictInt, StrictInt, StrictInt, StrictInt]], Field(min_length=1, max_length=4096)]
    market: str | None = None
    use_initial: bool = False

    @field_validator("docids")
    @classmethod
    def validate_docids(cls, values: list[PositionalID]) -> list[PositionalID]:
        for value in values:
            validate_docid(value)
        return values


class KeywordResult(BaseModel):
    docid: PositionalID
    keywords: list[str]
    status: str


class LookupResponse(BaseModel):
    mapping_version: str
    results: list[KeywordResult]


class DocidKeywordPair(BaseModel):
    docid: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    keyword_ids: Annotated[list[StrictInt], Field(min_length=1)]

    @field_validator("docid")
    @classmethod
    def validate_docid(cls, value: PositionalID) -> PositionalID:
        validate_docid(value)
        return value


class DocidKeywordTextPair(BaseModel):
    docid: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    keywords: Annotated[list[str], Field(min_length=1)]

    @field_validator("docid")
    @classmethod
    def validate_docid(cls, value: PositionalID) -> PositionalID:
        validate_docid(value)
        return value


class UpdateMappingRequest(BaseModel):
    entries: Annotated[list[DocidKeywordPair], Field(min_length=1)]
    replace_all: bool = False


class DeltaMappingRequest(BaseModel):
    additions: list[DocidKeywordTextPair] = Field(default_factory=list)
    removals: list[DocidKeywordTextPair] = Field(default_factory=list)
    k_max: int = DOCID_MAX_KEYWORDS


class DeltaMappingIdsRequest(BaseModel):
    additions: list[DocidKeywordPair] = Field(default_factory=list)
    removals: list[DocidKeywordPair] = Field(default_factory=list)
    k_max: int = DOCID_MAX_KEYWORDS


class ResolveKeywordsRequest(BaseModel):
    keywords: list[str]


class ClaimMappingRequest(BaseModel):
    owner: str
    checkpoint: str | None = None


class SaveMappingRequest(BaseModel):
    output_dir: str


def build_mapping_app(
    mapping: str | Path,
    *,
    mapping_cls: type[ResidentMapping],
    api_version: int,
) -> FastAPI:
    path = absolute_path(mapping)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.mapping = mapping_cls(path)
        gc.collect()
        gc.freeze()
        gc.set_threshold(100_000, 500, 500)
        yield
        del app.state.mapping

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        loaded = app.state.mapping
        return {
            "status": "ready",
            "mapping_version": loaded.version,
            "id_count": loaded.id_count,
            "rows": loaded.rows,
            "bytes": loaded.byte_count,
            "service_config": {"api_version": api_version, "mapping": str(path)},
        }

    @app.get("/runs/{owner}/health")
    def run_health(owner: str) -> dict[str, Any]:
        try:
            with app.state.mapping.access(owner):
                return health()
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/claim")
    def claim(request: ClaimMappingRequest) -> None:
        try:
            app.state.mapping.claim(request.owner, request.checkpoint)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/release")
    def release(request: ClaimMappingRequest) -> None:
        try:
            app.state.mapping.release(request.owner)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/lookup", response_model=LookupResponse)
    @app.post("/runs/{owner}/lookup", response_model=LookupResponse)
    def lookup(request: LookupRequest, http_request: Request) -> LookupResponse:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.lookup(
                    request.docids,
                    market=request.market,
                    use_initial=request.use_initial,
                )
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/runs/{owner}/resolve_keywords")
    def resolve_keywords(request: ResolveKeywordsRequest, http_request: Request) -> dict[str, list[int]]:
        try:
            with app.state.mapping.access(http_request.path_params["owner"]):
                return {"keyword_ids": app.state.mapping.resolve_keywords(request.keywords)}
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/update_delta")
    @app.post("/runs/{owner}/update_delta")
    def update_delta(request: DeltaMappingRequest, http_request: Request) -> dict[str, int]:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.apply_delta(request)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/runs/{owner}/update_delta_ids")
    def update_delta_ids(request: DeltaMappingIdsRequest, http_request: Request) -> dict[str, int]:
        try:
            with app.state.mapping.access(http_request.path_params["owner"]):
                return app.state.mapping.apply_id_delta(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/update")
    @app.post("/runs/{owner}/update")
    def update(request: UpdateMappingRequest, http_request: Request) -> dict[str, int]:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.update_docid_mappings(request.entries, replace_all=request.replace_all)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/save")
    @app.post("/runs/{owner}/save")
    def save(request: SaveMappingRequest, http_request: Request) -> dict[str, Any]:
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.save_docid_mappings(request.output_dir)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    return app


def run_mapping_cli(create_app_fn: Any, doc: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30488)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app_fn(args.mapping), host=args.host, port=args.port, workers=1)

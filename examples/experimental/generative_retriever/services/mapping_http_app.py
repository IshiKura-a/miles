"""FastAPI HTTP service wrapper around ResidentMapping."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
import uvicorn

from examples.experimental.generative_retriever.utils.paths import absolute_path

if TYPE_CHECKING:
    from examples.experimental.generative_retriever.services.mapping_service import (
        ClaimMappingRequest,
        DeltaMappingIdsRequest,
        DeltaMappingRequest,
        LookupRequest,
        LookupResponse,
        ResidentMapping,
        ResolveKeywordsRequest,
        SaveMappingRequest,
        UpdateMappingRequest,
    )


def build_mapping_app(
    mapping: str | Path,
    *,
    mapping_cls: type[ResidentMapping],
    api_version: int,
    claim_req_cls: type[ClaimMappingRequest],
    lookup_req_cls: type[LookupRequest],
    lookup_resp_cls: type[LookupResponse],
    resolve_req_cls: type[ResolveKeywordsRequest],
    delta_req_cls: type[DeltaMappingRequest],
    delta_ids_req_cls: type[DeltaMappingIdsRequest],
    update_req_cls: type[UpdateMappingRequest],
    save_req_cls: type[SaveMappingRequest],
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
    def claim(request: claim_req_cls) -> None:  # type: ignore[valid-type]
        try:
            app.state.mapping.claim(request.owner, request.checkpoint)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/release")
    def release(request: claim_req_cls) -> None:  # type: ignore[valid-type]
        try:
            app.state.mapping.release(request.owner)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/lookup", response_model=lookup_resp_cls)
    @app.post("/runs/{owner}/lookup", response_model=lookup_resp_cls)
    def lookup(request: lookup_req_cls, http_request: Request) -> LookupResponse:  # type: ignore[valid-type]
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
    def resolve_keywords(request: resolve_req_cls, http_request: Request) -> dict[str, list[int]]:  # type: ignore[valid-type]
        try:
            with app.state.mapping.access(http_request.path_params["owner"]):
                return {"keyword_ids": app.state.mapping.resolve_keywords(request.keywords)}
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/update_delta")
    @app.post("/runs/{owner}/update_delta")
    def update_delta(request: delta_req_cls, http_request: Request) -> dict[str, int]:  # type: ignore[valid-type]
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.apply_delta(request)
        except (ValueError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/runs/{owner}/update_delta_ids")
    def update_delta_ids(request: delta_ids_req_cls, http_request: Request) -> dict[str, int]:  # type: ignore[valid-type]
        try:
            with app.state.mapping.access(http_request.path_params["owner"]):
                return app.state.mapping.apply_id_delta(request)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/update")
    @app.post("/runs/{owner}/update")
    def update(request: update_req_cls, http_request: Request) -> dict[str, int]:  # type: ignore[valid-type]
        try:
            with app.state.mapping.access(http_request.path_params.get("owner")):
                return app.state.mapping.update_docid_mappings(request.entries, replace_all=request.replace_all)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/save")
    @app.post("/runs/{owner}/save")
    def save(request: save_req_cls, http_request: Request) -> dict[str, Any]:  # type: ignore[valid-type]
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

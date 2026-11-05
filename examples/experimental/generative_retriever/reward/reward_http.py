"""HTTP transport for idempotent generative-retriever reward requests."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 0.5
_RETRYABLE_TRANSPORT = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
)


def _should_retry(error: httpx.HTTPError, attempt: int, url: str) -> bool:
    retryable = isinstance(error, _RETRYABLE_TRANSPORT) or (
        isinstance(error, httpx.HTTPStatusError) and error.response.status_code in (502, 503)
    )
    if retryable:
        if attempt == _MAX_ATTEMPTS:
            logger.error("GR reward request to %s failed after %d attempts: %r", url, attempt, error)
        else:
            logger.warning(
                "GR reward request to %s failed on attempt %d/%d: %r; retrying the same payload",
                url, attempt, _MAX_ATTEMPTS, error,
            )
    return retryable and attempt < _MAX_ATTEMPTS


def _json_object(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("GR reward endpoint must return a JSON object")
    return payload


async def post_json_async(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(
        timeout=timeout,
        limits=httpx.Limits(max_keepalive_connections=0),
        trust_env=False,
    ) as client:
        attempt = 1
        while True:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return _json_object(response)
            except httpx.HTTPError as error:
                if not _should_retry(error, attempt, url):
                    raise
                await asyncio.sleep(_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
                attempt += 1


def post_json_sync(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    attempt = 1
    while True:
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return _json_object(response)
        except httpx.HTTPError as error:
            if not _should_retry(error, attempt, url):
                raise
            time.sleep(_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
            attempt += 1

"""HTTP transport and async request batching for generative-retriever services."""

from __future__ import annotations

import abc
import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import time
from typing import Any, Generic, TypeVar

import httpx

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 0.5
_DEFAULT_BATCH_WAIT_SECONDS = 0.01
_RETRYABLE_TRANSPORT = (
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.TimeoutException,
)

RequestT = TypeVar("RequestT")
ResultT = TypeVar("ResultT")


@dataclass(slots=True)
class PendingBatchRequest(Generic[RequestT, ResultT]):
    payload: RequestT
    weight: int
    future: asyncio.Future[ResultT]


class AsyncRequestBatcher(abc.ABC, Generic[RequestT, ResultT]):
    """Coalesce weighted async requests into bounded batches with zero-wait full-batch dispatch."""

    def __init__(
        self,
        *,
        max_batch_size: int,
        concurrency: int,
        batch_wait_seconds: float = _DEFAULT_BATCH_WAIT_SECONDS,
    ) -> None:
        if max_batch_size <= 0 or concurrency <= 0:
            raise ValueError("Batch size and concurrency must be positive")
        if batch_wait_seconds < 0:
            raise ValueError("Batch wait time must be non-negative")
        self._max_batch_size = max_batch_size
        self._batch_wait_seconds = batch_wait_seconds
        self._slots = asyncio.Semaphore(concurrency)
        self._requests: set[asyncio.Task[None]] = set()
        self._pending: deque[PendingBatchRequest[RequestT, ResultT]] = deque()
        self._pending_weight = 0
        self._full_batch_ready = asyncio.Event()
        self._flush_task: asyncio.Task[None] | None = None

    async def submit_weighted(self, payload: RequestT, *, weight: int) -> ResultT:
        if not 0 < weight <= self._max_batch_size:
            raise ValueError(f"Request weight must be in 1..{self._max_batch_size}, got {weight}")
        future = asyncio.get_running_loop().create_future()
        self._pending.append(PendingBatchRequest(payload=payload, weight=weight, future=future))
        self._pending_weight += weight
        if self._pending_weight >= self._max_batch_size:
            self._full_batch_ready.set()
        if self._flush_task is None:
            self._flush_task = asyncio.create_task(self.flush())
        return await future

    def take_batch(self) -> list[PendingBatchRequest[RequestT, ResultT]]:
        batch: list[PendingBatchRequest[RequestT, ResultT]] = []
        total_weight = 0
        while self._pending:
            next_request = self._pending[0]
            if batch and total_weight + next_request.weight > self._max_batch_size:
                break
            request = self._pending.popleft()
            self._pending_weight -= request.weight
            batch.append(request)
            total_weight += request.weight
        return batch

    async def wait_for_batch_window(self) -> None:
        if self._pending_weight >= self._max_batch_size or self._batch_wait_seconds == 0:
            return
        self._full_batch_ready.clear()
        if self._pending_weight >= self._max_batch_size:
            return
        try:
            await asyncio.wait_for(
                self._full_batch_ready.wait(),
                timeout=self._batch_wait_seconds,
            )
        except TimeoutError:
            pass

    async def flush(self) -> None:
        try:
            while self._pending:
                await self.wait_for_batch_window()
                if not self._pending:
                    break
                await self._slots.acquire()
                if not self._pending:
                    self._slots.release()
                    break
                requests = self.take_batch()
                task = asyncio.create_task(self.dispatch(requests))
                self._requests.add(task)
                task.add_done_callback(self._requests.discard)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.create_task(self.flush())

    async def dispatch(
        self,
        requests: list[PendingBatchRequest[RequestT, ResultT]],
    ) -> None:
        try:
            await self.execute_batch(requests)
        except asyncio.CancelledError:
            for request in requests:
                request.future.cancel()
            raise
        except Exception as error:
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(error)
        finally:
            self._slots.release()

    @abc.abstractmethod
    async def execute_batch(
        self,
        requests: list[PendingBatchRequest[RequestT, ResultT]],
    ) -> None:
        raise NotImplementedError


def should_retry(error: httpx.HTTPError, attempt: int, url: str) -> bool:
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


def json_object(response: httpx.Response) -> dict[str, Any]:
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("GR reward endpoint must return a JSON object")
    return payload


from weakref import WeakKeyDictionary
_ASYNC_CLIENTS: WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = WeakKeyDictionary()


def _get_async_client(timeout: float) -> httpx.AsyncClient:
    loop = asyncio.get_running_loop()
    client = _ASYNC_CLIENTS.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256),
            trust_env=False,
        )
        _ASYNC_CLIENTS[loop] = client
    return client


async def post_json_async(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    client = _get_async_client(timeout)
    attempt = 1
    while True:
        try:
            response = await client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return json_object(response)
        except httpx.HTTPError as error:
            if not should_retry(error, attempt, url):
                raise
            await asyncio.sleep(_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
            attempt += 1


def post_json_sync(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    attempt = 1
    while True:
        try:
            response = client.post(url, json=payload)
            response.raise_for_status()
            return json_object(response)
        except httpx.HTTPError as error:
            if not should_retry(error, attempt, url):
                raise
            time.sleep(_RETRY_DELAY_SECONDS * 2 ** (attempt - 1))
            attempt += 1

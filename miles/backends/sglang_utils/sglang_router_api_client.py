import asyncio
import dataclasses
from http import HTTPStatus
import logging
import time
from urllib.parse import quote

import httpx
import sglang_router
from packaging.version import parse

from miles.utils.http_utils import GeneralHttpClientProvider

logger = logging.getLogger(__name__)

ROUTER_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
WORKER_ACTIVATION_TIMEOUT_SECONDS = 60.0
WORKER_ACTIVATION_POLL_INTERVAL_SECONDS = 0.2


def use_legacy_router_api(args) -> bool:
    return parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router


async def _wait_for_worker_activation(router_url: str, worker_url: str) -> None:
    target_url = worker_url.rstrip("/")
    deadline = time.monotonic() + WORKER_ACTIVATION_TIMEOUT_SECONDS
    while True:
        response = await GeneralHttpClientProvider.client().get(
            f"{router_url}/workers",
            timeout=ROUTER_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        workers = response.json().get("workers", [])
        if any(isinstance(w, dict) and str(w.get("url", "")).rstrip("/") == target_url for w in workers):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {WORKER_ACTIVATION_TIMEOUT_SECONDS}s waiting for SGLang router "
                f"{router_url} to activate worker {worker_url}"
            )
        await asyncio.sleep(WORKER_ACTIVATION_POLL_INTERVAL_SECONDS)


@dataclasses.dataclass(frozen=True)
class SGLangRouterApiClient:
    router_url: str

    async def add_worker(
        self,
        worker_url: str,
        worker_type: str,
        use_legacy_api: bool,
        bootstrap_port: int | None = None,
    ):
        if use_legacy_api:
            assert worker_type == "regular", "pd disaggregation is not supported in old router or miles router."
            response = await GeneralHttpClientProvider.client().post(
                f"{self.router_url}/add_worker?url={worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            payload = {
                "url": worker_url,
                "worker_type": worker_type,
            }
            if worker_type == "prefill":
                payload["bootstrap_port"] = bootstrap_port
            response = await GeneralHttpClientProvider.client().post(
                f"{self.router_url}/workers",
                json=payload,
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        if not use_legacy_api and response.status_code == HTTPStatus.ACCEPTED:
            await _wait_for_worker_activation(self.router_url, worker_url)

    async def remove_worker(self, worker_url: str, use_legacy_api: bool):
        response = None
        if use_legacy_api:
            response = await GeneralHttpClientProvider.client().post(
                f"{self.router_url}/remove_worker?url={worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        elif parse(sglang_router.__version__) < parse("0.3.0"):
            quoted_worker_url = quote(worker_url, safe="")
            response = await GeneralHttpClientProvider.client().delete(
                f"{self.router_url}/workers/{quoted_worker_url}",
                timeout=ROUTER_REQUEST_TIMEOUT,
            )
        else:
            try:
                all_workers = (
                    await GeneralHttpClientProvider.client().get(
                        f"{self.router_url}/workers",
                        timeout=ROUTER_REQUEST_TIMEOUT,
                    )
                ).json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        worker_id = worker["id"]
                        response = await GeneralHttpClientProvider.client().delete(
                            f"{self.router_url}/workers/{worker_id}",
                            timeout=ROUTER_REQUEST_TIMEOUT,
                        )
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        if response is not None:
            response.raise_for_status()

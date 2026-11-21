"""Immutable runtime configuration and annealing schedules for Generative Retriever RL."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import asdict, dataclass
import json
import math
import os
from typing import Literal, cast

DocidDenomMode = Literal["sqrt_floor", "unit", "linear_floor"]
Doc2kMode = Literal["relevance_guided", "legacy", "disabled"]
UniqueAnnealMode = Literal["none", "off", "linear", "cosine"]
RewardModeLiteral = Literal["discrete_relevance_unique", "continuous_relevance_unique"]

DOCID_DENOM_MODES: tuple[DocidDenomMode, ...] = ("sqrt_floor", "unit", "linear_floor")
DOC2K_MODES: tuple[Doc2kMode, ...] = ("relevance_guided", "legacy", "disabled")
UNIQUE_ANNEAL_MODES: tuple[UniqueAnnealMode, ...] = ("none", "off", "linear", "cosine")

GR_CONFIG_ENV_VAR = "GR_CONFIG_JSON"
DEFAULT_STEP_TEMPERATURES: tuple[float, float, float, float] = (0.2, 0.25, 0.6, 0.8)
DEFAULT_SGLANG_WATCHDOG_TIMEOUT_SECONDS = 600
DEFAULT_HBM_KW_MEMORY_FRACTION = 0.20

RAY_GRPC_KEEPALIVE_ENV: dict[str, str] = {
    "RAY_grpc_client_idle_timeout_ms": "36000000",
    "RAY_grpc_client_keepalive_time_ms": "30000",
    "RAY_grpc_client_keepalive_timeout_ms": "600000",
    "RAY_grpc_keepalive_time_ms": "30000",
    "RAY_grpc_keepalive_timeout_ms": "600000",
    "RAY_timeout_ms_task_wait_for_death_info": "600000",
    "RAY_core_worker_rpc_server_reconnect_timeout_max_s": "600",
}


def apply_ray_keepalive_env(target_env: MutableMapping[str, str] | None = None) -> None:
    """Populate Ray transport-level HTTP/2 keepalive settings without import-time side effects."""
    env = os.environ if target_env is None else target_env
    for key, value in RAY_GRPC_KEEPALIVE_ENV.items():
        env.setdefault(key, value)


def parse_step_temperatures(raw: str | Sequence[float], *, code_num: int = 4) -> tuple[float, ...]:
    """Parse either a single uniform temperature (e.g. ``'0.4'``) or per-depth values (``'0.2,0.25,0.6,0.8'``)."""
    if isinstance(raw, str):
        parts = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        parts = [float(item) for item in raw]
    if len(parts) == 1:
        parts = parts * code_num
    if len(parts) != code_num or any(val <= 0.0 for val in parts):
        raise ValueError(f"Expected 1 or {code_num} positive step temperatures, got {raw!r}")
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class GRConfig:
    """Single source of truth for Generative Retriever algorithm and systems settings."""

    reward_mode: RewardModeLiteral = "discrete_relevance_unique"
    step_temperatures: tuple[float, ...] = DEFAULT_STEP_TEMPERATURES
    docid_denom_mode: DocidDenomMode = "sqrt_floor"
    docid_reward_floor: int = 10
    causal_tree_credit: bool = True
    doc2k_mode: Doc2kMode = "relevance_guided"
    unique_anneal_mode: UniqueAnnealMode = "none"
    unique_weight_start: float = 2.0
    unique_weight_end: float = 0.0
    unique_anneal_warmup_ratio: float = 0.0
    local_coloc_rollout: bool = True
    local_rollout_batch_size: int = 125
    compact_train_ipc: bool = True
    grouped_cross_attention: bool = True
    tree_query_chunk_size: int = 128
    hbm_kw_cache_enabled: bool = True
    hbm_kw_memory_fraction: float = DEFAULT_HBM_KW_MEMORY_FRACTION
    hbm_kw_explicit_gib: float | None = None

    def __post_init__(self) -> None:
        normalized_temps = parse_step_temperatures(self.step_temperatures)
        object.__setattr__(self, "step_temperatures", normalized_temps)
        if self.docid_denom_mode not in DOCID_DENOM_MODES:
            raise ValueError(f"Unsupported docid_denom_mode: {self.docid_denom_mode!r}")
        if self.doc2k_mode not in DOC2K_MODES:
            raise ValueError(f"Unsupported doc2k_mode: {self.doc2k_mode!r}")
        if self.unique_anneal_mode not in UNIQUE_ANNEAL_MODES:
            raise ValueError(f"Unsupported unique_anneal_mode: {self.unique_anneal_mode!r}")
        if self.unique_weight_start < 0.0 or self.unique_weight_end < 0.0:
            raise ValueError("unique_weight_start and unique_weight_end must be non-negative")
        if not (0.0 <= self.unique_anneal_warmup_ratio < 1.0):
            raise ValueError("unique_anneal_warmup_ratio must be in [0.0, 1.0)")
        if not (0.0 <= self.hbm_kw_memory_fraction < 1.0):
            raise ValueError("hbm_kw_memory_fraction must be in [0.0, 1.0)")

    @property
    def doc2k_disabled(self) -> bool:
        return self.doc2k_mode == "disabled"

    @property
    def doc2k_relevance_guided(self) -> bool:
        return self.doc2k_mode == "relevance_guided"

    @property
    def pm_prune_only(self) -> bool:
        return self.doc2k_mode == "relevance_guided"

    def resolve_unique_weight(self, rollout_id: int, num_rollout: int) -> float:
        """Compute exact step ``unique_bonus`` weight from ``rollout_id`` (resume-invariant)."""
        if self.unique_anneal_mode == "off":
            return 0.0
        if self.unique_anneal_mode == "none":
            return float(self.unique_weight_start)
        total_spans = max(1, int(num_rollout) - 1)
        progress = min(1.0, max(0.0, float(rollout_id) / float(total_spans)))
        if progress <= self.unique_anneal_warmup_ratio:
            return float(self.unique_weight_start)
        decay_progress = (progress - self.unique_anneal_warmup_ratio) / (1.0 - self.unique_anneal_warmup_ratio)
        if self.unique_anneal_mode == "linear":
            factor = 1.0 - decay_progress
        elif self.unique_anneal_mode == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        else:
            raise ValueError(f"Unsupported unique_anneal_mode: {self.unique_anneal_mode!r}")
        return float(self.unique_weight_end + (self.unique_weight_start - self.unique_weight_end) * factor)

    def to_json(self) -> str:
        data = asdict(self)
        data["step_temperatures"] = list(self.step_temperatures)
        return json.dumps(data, sort_keys=True)

    def to_env_dict(self) -> dict[str, str]:
        return {GR_CONFIG_ENV_VAR: self.to_json()}

    @classmethod
    def from_json(cls, payload: str) -> GRConfig:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("GR_CONFIG_JSON must decode to a JSON object")
        if "step_temperatures" in raw:
            raw["step_temperatures"] = tuple(float(x) for x in raw["step_temperatures"])
        return cls(**raw)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GRConfig:
        env = os.environ if environ is None else environ
        payload = env.get(GR_CONFIG_ENV_VAR, "").strip()
        if payload:
            return cls.from_json(payload)
        reward_mode = cast(
            RewardModeLiteral,
            env.get("GR_REWARD_MODE", "discrete_relevance_unique"),
        )
        doc2k_mode: Doc2kMode = "disabled" if env.get("GR_DISABLE_DOC2K") == "1" else "relevance_guided"
        return cls(reward_mode=reward_mode, doc2k_mode=doc2k_mode)

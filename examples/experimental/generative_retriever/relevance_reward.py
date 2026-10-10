"""Approved PM/SM relevance and novelty reward for valid keyword candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

RewardMode = Literal["continuous_relevance_unique", "discrete_relevance_unique"]
REWARD_MODES: tuple[RewardMode, ...] = ("continuous_relevance_unique", "discrete_relevance_unique")

PM_THRESHOLD = 0.58
SM_THRESHOLD = 0.82
BASELINE_TOP_K = 500
UNIQUE_WEIGHT = 1.0


@dataclass(frozen=True)
class RewardBreakdown:
    pm_score: float
    sm_score: float
    normalized_pm: float
    normalized_sm: float
    relevance: float
    is_unique: bool
    unique_bonus: float
    reward: float
    mode: RewardMode = "continuous_relevance_unique"


def _normalize_score(score: float, threshold: float, name: str) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError(f"{name} must be a numeric probability")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {score!r}")
    return max(0.0, (score - threshold) / (1.0 - threshold))


def calculate_reward(
    *, pm_score: float, sm_score: float, is_unique: bool,
    mode: RewardMode = "continuous_relevance_unique",
) -> RewardBreakdown:
    """Combine PM/SM with keyword novelty against the filtered baseline keyword set."""
    if not isinstance(is_unique, bool):
        raise TypeError("is_unique must be a boolean")
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    normalized_pm = _normalize_score(pm_score, PM_THRESHOLD, "PM score")
    normalized_sm = _normalize_score(sm_score, SM_THRESHOLD, "SM score")
    if mode == "continuous_relevance_unique":
        relevance = normalized_pm + normalized_sm
        unique_bonus = UNIQUE_WEIGHT * relevance if is_unique else 0.0
    else:
        relevance = float(int(pm_score >= PM_THRESHOLD) + int(sm_score >= SM_THRESHOLD))
        unique_bonus = 1.0 if relevance == 2.0 and is_unique else 0.0
    return RewardBreakdown(
        pm_score=float(pm_score),
        sm_score=float(sm_score),
        normalized_pm=normalized_pm,
        normalized_sm=normalized_sm,
        relevance=relevance,
        is_unique=is_unique,
        unique_bonus=unique_bonus,
        reward=relevance + unique_bonus,
        mode=mode,
    )


def build_baseline_set(keyword_scores: Mapping[str, tuple[float, float]]) -> frozenset[str]:
    """Keep keywords meeting both PM and SM thresholds."""
    qualified = set()
    for keyword, (pm_score, sm_score) in keyword_scores.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Baseline keyword must be a non-empty string")
        calculate_reward(pm_score=pm_score, sm_score=sm_score, is_unique=False)
        if pm_score >= PM_THRESHOLD and sm_score >= SM_THRESHOLD:
            qualified.add(keyword)
    return frozenset(qualified)


def calculate_keyword_rewards(
    keyword_scores: Mapping[str, tuple[float, float]], baseline: frozenset[str],
    *, mode: RewardMode = "continuous_relevance_unique",
) -> dict[str, RewardBreakdown]:
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    results = {}
    for keyword, (pm_score, sm_score) in keyword_scores.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Candidate keyword must be a non-empty string")
        results[keyword] = calculate_reward(
            pm_score=pm_score, sm_score=sm_score, is_unique=keyword not in baseline,
            mode=mode,
        )
    return results


def calculate_docid_reward(
    keyword_scores: Mapping[str, tuple[float, float]], baseline: frozenset[str],
    *, mode: RewardMode = "continuous_relevance_unique",
) -> float:
    """Average rewards over distinct keyword texts belonging to one doc ID."""
    if not keyword_scores:
        raise ValueError("Doc ID reward requires at least one keyword")
    rewards = calculate_keyword_rewards(keyword_scores, baseline, mode=mode)
    return math.fsum(result.reward for result in rewards.values()) / len(rewards)

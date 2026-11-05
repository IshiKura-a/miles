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
BASELINE_TOP_K = 16
UNIQUE_WEIGHT = 1.0


@dataclass(frozen=True)
class BaselineSets:
    pm: frozenset[str] = frozenset()
    sm: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RewardBreakdown:
    pm_score: float
    sm_score: float
    normalized_pm: float
    normalized_sm: float
    pm_pass: float
    sm_pass: float
    pm_relevance: float
    sm_relevance: float
    relevance: float
    relevance_quality: float
    is_pm_unique: bool
    is_sm_unique: bool
    is_unique: bool
    relevance_eligible: bool
    pm_unique_eligible: bool
    sm_unique_eligible: bool
    unique_eligible: bool
    pm_unique_bonus: float
    sm_unique_bonus: float
    unique_bonus: float
    discrete_base: float
    continuous_base: float
    discrete_total: float
    continuous_total: float
    reward: float
    mode: RewardMode = "discrete_relevance_unique"


def _normalize_score(score: float, threshold: float, name: str) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError(f"{name} must be a numeric probability")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {score!r}")
    return min(score / threshold, 1.0)


def calculate_reward(
    *,
    pm_score: float,
    sm_score: float,
    is_pm_unique: bool,
    is_sm_unique: bool,
    mode: RewardMode = "discrete_relevance_unique",
) -> RewardBreakdown:
    """Combine PM/SM with per-head keyword novelty against the baseline sets."""
    if not isinstance(is_pm_unique, bool) or not isinstance(is_sm_unique, bool):
        raise TypeError("is_pm_unique and is_sm_unique must be booleans")
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    normalized_pm = _normalize_score(pm_score, PM_THRESHOLD, "PM score")
    normalized_sm = _normalize_score(sm_score, SM_THRESHOLD, "SM score")
    pm_pass = float(pm_score >= PM_THRESHOLD)
    sm_pass = float(sm_score >= SM_THRESHOLD)
    discrete_base = pm_pass + sm_pass
    continuous_base = normalized_pm + normalized_sm
    relevance_quality = continuous_base / 2.0
    relevance_eligible = bool(pm_pass or sm_pass)
    pm_unique_eligible = is_pm_unique and bool(pm_pass)
    sm_unique_eligible = is_sm_unique and bool(sm_pass)
    unique_eligible = pm_unique_eligible or sm_unique_eligible
    combined_unique = is_pm_unique and is_sm_unique
    pm_unique_bonus = 0.5 * UNIQUE_WEIGHT * relevance_quality if pm_unique_eligible else 0.0
    sm_unique_bonus = 0.5 * UNIQUE_WEIGHT * relevance_quality if sm_unique_eligible else 0.0
    unique_bonus = pm_unique_bonus + sm_unique_bonus
    discrete_total = discrete_base + unique_bonus
    continuous_total = continuous_base + unique_bonus
    if mode == "continuous_relevance_unique":
        pm_relevance = normalized_pm
        sm_relevance = normalized_sm
        relevance = continuous_base
        reward = continuous_total
    else:
        pm_relevance = pm_pass
        sm_relevance = sm_pass
        relevance = discrete_base
        reward = discrete_total
    return RewardBreakdown(
        pm_score=float(pm_score),
        sm_score=float(sm_score),
        normalized_pm=normalized_pm,
        normalized_sm=normalized_sm,
        pm_pass=pm_pass,
        sm_pass=sm_pass,
        pm_relevance=pm_relevance,
        sm_relevance=sm_relevance,
        relevance=relevance,
        relevance_quality=relevance_quality,
        is_pm_unique=is_pm_unique,
        is_sm_unique=is_sm_unique,
        is_unique=combined_unique,
        relevance_eligible=relevance_eligible,
        pm_unique_eligible=pm_unique_eligible,
        sm_unique_eligible=sm_unique_eligible,
        unique_eligible=unique_eligible,
        pm_unique_bonus=pm_unique_bonus,
        sm_unique_bonus=sm_unique_bonus,
        unique_bonus=unique_bonus,
        discrete_base=discrete_base,
        continuous_base=continuous_base,
        discrete_total=discrete_total,
        continuous_total=continuous_total,
        reward=reward,
        mode=mode,
    )


def build_baseline_set(keyword_scores: Mapping[str, tuple[float, float]]) -> BaselineSets:
    """Keep per-head baseline keyword sets meeting PM and SM thresholds."""
    pm_qualified = set()
    sm_qualified = set()
    for keyword, (pm_score, sm_score) in keyword_scores.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Baseline keyword must be a non-empty string")
        calculate_reward(
            pm_score=pm_score,
            sm_score=sm_score,
            is_pm_unique=False,
            is_sm_unique=False,
        )
        if pm_score >= PM_THRESHOLD:
            pm_qualified.add(keyword)
        if sm_score >= SM_THRESHOLD:
            sm_qualified.add(keyword)
    return BaselineSets(pm=frozenset(pm_qualified), sm=frozenset(sm_qualified))


def calculate_keyword_rewards(
    keyword_scores: Mapping[str, tuple[float, float]],
    baseline: BaselineSets,
    *,
    mode: RewardMode = "discrete_relevance_unique",
) -> dict[str, RewardBreakdown]:
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    results = {}
    for keyword, (pm_score, sm_score) in keyword_scores.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Candidate keyword must be a non-empty string")
        results[keyword] = calculate_reward(
            pm_score=pm_score,
            sm_score=sm_score,
            is_pm_unique=keyword not in baseline.pm,
            is_sm_unique=keyword not in baseline.sm,
            mode=mode,
        )
    return results


def calculate_docid_reward(
    keyword_scores: Mapping[str, tuple[float, float]],
    baseline: BaselineSets,
    *,
    mode: RewardMode = "discrete_relevance_unique",
) -> float:
    """Average rewards over distinct keyword texts belonging to one doc ID."""
    if not keyword_scores:
        raise ValueError("Doc ID reward requires at least one keyword")
    rewards = calculate_keyword_rewards(keyword_scores, baseline, mode=mode)
    return math.fsum(result.reward for result in rewards.values()) / len(rewards)

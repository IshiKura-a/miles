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
DEFAULT_TRAIN_GROUP_SIZE = 512
DEFAULT_EVAL_GROUP_SIZE = 64
DEFAULT_EVAL_BATCH_SIZE = 1024
MAX_EVAL_BATCH_SIZE = 1024
MAX_DOCIDS_PER_QUERY = 512
RQ_CODE_NUM = 4
RQ_CODE_DIM = 256
DOCID_TOKEN_OFFSET = 2
QUERY_MAX_TOKENS = 16
PROMPT_DATASET_ROWS = 15_000_000
DEFAULT_REWARD_BATCH_SIZE = 4096
REWARD_TOWER_BATCH_SIZE = 8192
REWARD_PAIR_BATCH_SIZE = 32768
UNIQUE_WEIGHT = 2.0
DOCID_REWARD_FLOOR = 10
DOC2K_PRUNE_FLOOR = 32
DOCID_MAX_KEYWORDS = 150
DOC2K_MIN_GROUP_SAMPLES = 4
DOC2K_HEAD_RANK_CUTOFF = 64
DOC2K_MIN_HEAD_RANKS = 16
DOC2K_MAX_ADDS_PER_QUERY = 8
DOC2K_ADD_MIN_QUERIES = 10
DOC2K_PRUNE_MIN_QUERIES = 32
TREE_CREDIT_TOP_K = (8, 4, 2)
TREE_CREDIT_LEVEL_WEIGHT = 0.25


@dataclass(frozen=True, slots=True)
class BaselineSets:
    pm: frozenset[str] = frozenset()
    sm: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
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
    keyword_frequency: int = 1
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
    keyword_frequency: int = 1,
) -> RewardBreakdown:
    """Combine PM/SM with per-head keyword novelty against the baseline sets."""
    if not isinstance(is_pm_unique, bool) or not isinstance(is_sm_unique, bool):
        raise TypeError("is_pm_unique and is_sm_unique must be booleans")
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    if isinstance(keyword_frequency, bool) or not isinstance(keyword_frequency, int) or keyword_frequency < 1:
        raise ValueError(f"keyword_frequency must be a positive integer, got {keyword_frequency!r}")
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
        raw_reward = continuous_total
    else:
        pm_relevance = pm_pass
        sm_relevance = sm_pass
        relevance = discrete_base
        raw_reward = discrete_total
    reward = raw_reward / float(keyword_frequency)
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
        keyword_frequency=keyword_frequency,
        mode=mode,
    )


def build_baseline_set(keyword_scores: Mapping[str, tuple[float, float]]) -> BaselineSets:
    """Keep per-head baseline keyword sets meeting PM and SM thresholds."""
    pm_qualified = set()
    sm_qualified = set()
    for keyword, (pm_score, sm_score) in keyword_scores.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Baseline keyword must be a non-empty string")
        _normalize_score(pm_score, PM_THRESHOLD, "PM score")
        _normalize_score(sm_score, SM_THRESHOLD, "SM score")
        if pm_score >= PM_THRESHOLD:
            pm_qualified.add(keyword)
        if sm_score >= SM_THRESHOLD:
            sm_qualified.add(keyword)
    return BaselineSets(pm=frozenset(pm_qualified), sm=frozenset(sm_qualified))


def score_and_summarize_docid(
    keywords: list[str],
    pair_scores: Mapping[tuple[str, str], tuple[float, float]],
    query: str,
    baseline: BaselineSets,
    *,
    mode: RewardMode = "discrete_relevance_unique",
    keyword_frequencies: Mapping[str, int] | None = None,
    k_floor: int = DOCID_REWARD_FLOOR,
) -> tuple[float, dict[str, float], list[str], list[str]]:
    """Compute docID reward, summary metrics, and doc2k edge sets without per-keyword object allocation."""
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    if not keywords:
        raise ValueError("Doc ID reward requires at least one keyword")
    n = float(len(keywords))
    base_pm = baseline.pm
    base_sm = baseline.sm
    is_cont = mode == "continuous_relevance_unique"

    sum_pm = 0.0
    sum_sm = 0.0
    sum_pm_pass = 0.0
    sum_sm_pass = 0.0
    sum_norm_pm = 0.0
    sum_norm_sm = 0.0
    sum_rel_qual = 0.0
    sum_pm_ub = 0.0
    sum_sm_ub = 0.0
    sum_ub = 0.0
    sum_disc_base = 0.0
    sum_cont_base = 0.0
    sum_disc_tot = 0.0
    sum_cont_tot = 0.0
    sum_reward = 0.0
    cnt_pm_uniq = 0
    cnt_sm_uniq = 0
    cnt_both_uniq = 0
    cnt_rel_elig = 0
    cnt_pm_u_elig = 0
    cnt_sm_u_elig = 0
    cnt_u_elig = 0
    sum_uniq_qual = 0.0
    cnt_any_uniq = 0

    passed_keywords: list[str] = []
    unique_good_keywords: list[str] = []

    for kw in keywords:
        pm_s, sm_s = pair_scores[(query, kw)]
        pm_f = float(pm_s)
        sm_f = float(sm_s)
        norm_pm = pm_f / PM_THRESHOLD if pm_f < PM_THRESHOLD else 1.0
        norm_sm = sm_f / SM_THRESHOLD if sm_f < SM_THRESHOLD else 1.0
        p_ok = pm_f >= PM_THRESHOLD
        s_ok = sm_f >= SM_THRESHOLD
        pm_p = 1.0 if p_ok else 0.0
        sm_p = 1.0 if s_ok else 0.0
        disc_b = pm_p + sm_p
        cont_b = norm_pm + norm_sm
        rel_q = cont_b * 0.5

        u_pm = kw not in base_pm
        u_sm = kw not in base_sm
        pm_u_el = u_pm and p_ok
        sm_u_el = u_sm and s_ok
        u_el = pm_u_el or sm_u_el
        rel_el = p_ok or s_ok

        pm_ub = (0.5 * UNIQUE_WEIGHT * rel_q) if pm_u_el else 0.0
        sm_ub = (0.5 * UNIQUE_WEIGHT * rel_q) if sm_u_el else 0.0
        ub = pm_ub + sm_ub
        disc_t = disc_b + ub
        cont_t = cont_b + ub
        raw_r = cont_t if is_cont else disc_t
        freq = max(1, int(keyword_frequencies.get(kw, 1))) if keyword_frequencies is not None else 1
        rew = raw_r / float(freq)

        sum_pm += pm_f
        sum_sm += sm_f
        sum_pm_pass += pm_p
        sum_sm_pass += sm_p
        sum_norm_pm += norm_pm
        sum_norm_sm += norm_sm
        sum_rel_qual += rel_q
        sum_pm_ub += pm_ub
        sum_sm_ub += sm_ub
        sum_ub += ub
        sum_disc_base += disc_b
        sum_cont_base += cont_b
        sum_disc_tot += disc_t
        sum_cont_tot += cont_t
        sum_reward += rew

        if u_pm:
            cnt_pm_uniq += 1
        if u_sm:
            cnt_sm_uniq += 1
        if u_pm and u_sm:
            cnt_both_uniq += 1
        if u_pm or u_sm:
            sum_uniq_qual += rel_q
            cnt_any_uniq += 1
        if rel_el:
            cnt_rel_elig += 1
            passed_keywords.append(kw)
        if p_ok and s_ok and u_pm and u_sm:
            unique_good_keywords.append(kw)
        if pm_u_el:
            cnt_pm_u_elig += 1
        if sm_u_el:
            cnt_sm_u_elig += 1
        if u_el:
            cnt_u_elig += 1

    mean_reward = sum_reward / n
    metrics = {
        "pm_score": sum_pm / n,
        "sm_score": sum_sm / n,
        "pm_pass": sum_pm_pass / n,
        "sm_pass": sum_sm_pass / n,
        "normalized_pm": sum_norm_pm / n,
        "normalized_sm": sum_norm_sm / n,
        "pm_relevance": (sum_norm_pm if is_cont else sum_pm_pass) / n,
        "sm_relevance": (sum_norm_sm if is_cont else sum_sm_pass) / n,
        "relevance_quality": sum_rel_qual / n,
        "pm_unique_bonus": sum_pm_ub / n,
        "sm_unique_bonus": sum_sm_ub / n,
        "unique_bonus": sum_ub / n,
        "discrete_base": sum_disc_base / n,
        "continuous_base": sum_cont_base / n,
        "discrete_total": sum_disc_tot / n,
        "continuous_total": sum_cont_tot / n,
        "reward": mean_reward,
        "pm_unique_keyword_rate": float(cnt_pm_uniq) / n,
        "sm_unique_keyword_rate": float(cnt_sm_uniq) / n,
        "unique_keyword_rate": float(cnt_both_uniq) / n,
        "relevance_eligible_rate": float(cnt_rel_elig) / n,
        "pm_unique_eligible_rate": float(cnt_pm_u_elig) / n,
        "sm_unique_eligible_rate": float(cnt_sm_u_elig) / n,
        "unique_eligible_rate": float(cnt_u_elig) / n,
        "unique_quality": (sum_uniq_qual / float(cnt_any_uniq)) if cnt_any_uniq else 0.0,
        "train_total": mean_reward,
    }
    docid_value = sum_reward / float(max(len(keywords), k_floor))
    return docid_value, metrics, passed_keywords, unique_good_keywords


def calculate_keyword_rewards(
    keyword_scores: Mapping[str, tuple[float, float]],
    baseline: BaselineSets,
    *,
    mode: RewardMode = "discrete_relevance_unique",
    keyword_frequencies: Mapping[str, int] | None = None,
) -> dict[str, RewardBreakdown]:
    if mode not in REWARD_MODES:
        raise ValueError(f"Unknown reward mode: {mode!r}")
    results = {}
    for keyword, (pm_score, sm_score) in keyword_scores.items():
        if not isinstance(keyword, str) or not keyword.strip():
            raise ValueError("Candidate keyword must be a non-empty string")
        freq = max(1, int(keyword_frequencies.get(keyword, 1))) if keyword_frequencies is not None else 1
        results[keyword] = calculate_reward(
            pm_score=pm_score,
            sm_score=sm_score,
            is_pm_unique=keyword not in baseline.pm,
            is_sm_unique=keyword not in baseline.sm,
            mode=mode,
            keyword_frequency=freq,
        )
    return results


def calculate_docid_reward(
    keyword_scores: Mapping[str, tuple[float, float]],
    baseline: BaselineSets,
    *,
    mode: RewardMode = "discrete_relevance_unique",
    keyword_frequencies: Mapping[str, int] | None = None,
    k_floor: int = DOCID_REWARD_FLOOR,
) -> float:
    """Aggregate rewards over distinct keyword texts belonging to one doc ID with floor denominator."""
    if not keyword_scores:
        raise ValueError("Doc ID reward requires at least one keyword")
    if k_floor <= 0:
        raise ValueError("k_floor must be positive")
    rewards = calculate_keyword_rewards(
        keyword_scores,
        baseline,
        mode=mode,
        keyword_frequencies=keyword_frequencies,
    )
    denominator = max(len(rewards), k_floor)
    return math.fsum(result.reward for result in rewards.values()) / float(denominator)

"""Build filtered baseline keyword sets from official constrained GR outputs.

Example:
    python -m examples.experimental.generative_retriever.build_baseline_rewards \
        --baseline-results /path/to/official-gr-constrained-reward-inputs.json \
        --mapping-url http://127.0.0.1:30488 \
        --relevance-source /path/to/checkpoint/code \
        --relevance-model /path/to/ForArcher/Large \
        --output-dir /path/to/baseline-rewards --max-queries 1
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.experimental.generative_retriever.id_text_mapping import lookup_service
from examples.experimental.generative_retriever.model_loading import require_directory
from examples.experimental.generative_retriever.relevance_model import (
    load_relevance_pipeline_factory,
    resolve_relevance_source,
)
from examples.experimental.generative_retriever.relevance_reward import (
    BASELINE_TOP_K, PM_THRESHOLD, SM_THRESHOLD, REWARD_MODES, RewardMode,
    build_baseline_set, calculate_docid_reward, calculate_keyword_rewards,
)

logger = logging.getLogger(__name__)


def _build_query(
    query_record, pipeline, mapping_url: str, mapping_version: str, destination: Path,
    reward_mode: RewardMode = "discrete_relevance_unique",
):
    query = query_record["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Baseline query must be a non-empty string")
    ids = [tuple(code) for code in query_record["baseline_codes"]]
    if len(ids) != BASELINE_TOP_K or len(set(ids)) != BASELINE_TOP_K:
        raise ValueError(f"Official baseline must contain exactly {BASELINE_TOP_K} distinct doc IDs")
    mapping = lookup_service(mapping_url, ids, expected_mapping_version=mapping_version)
    keyword_ids = {}
    for docid, keywords in mapping.items():
        for keyword in keywords:
            keyword_ids.setdefault(keyword, []).append(docid)
    keywords = list(keyword_ids)
    logger.info("Scoring query %r: %d IDs, %d distinct keywords", query, len(ids), len(keywords))
    query_features = pipeline.encode_query([query])
    keyword_features = pipeline.encode_keyword(keywords)
    scores = pipeline.cross_score(
        np.repeat(query_features, len(keywords), axis=0), keyword_features,
    )
    if scores.shape != (len(keywords), 4):
        raise RuntimeError(f"Unexpected relevance score shape: {scores.shape}")
    if not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise RuntimeError("Relevance model returned invalid probabilities")
    keyword_scores = {
        keyword: (float(score[1]), float(score[2]))
        for keyword, score in zip(keywords, scores, strict=True)
    }
    baseline = build_baseline_set(keyword_scores)
    baseline_union = baseline.pm | baseline.sm
    rewards = calculate_keyword_rewards(keyword_scores, baseline, mode=reward_mode)
    rows = []
    for keyword in keywords:
        rows.append({
            "query": query, "keyword": keyword, "docids": keyword_ids[keyword],
            "in_baseline_pm_set": keyword in baseline.pm,
            "in_baseline_sm_set": keyword in baseline.sm,
            "in_baseline_set": keyword in baseline_union,
            **asdict(rewards[keyword]),
        })
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "keyword_rewards.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (destination / "docid_rewards.jsonl").open("w", encoding="utf-8") as output:
        for docid, texts in mapping.items():
            output.write(json.dumps({
                "query": query, "docid": docid, "keyword_count": len(texts),
                "mode": reward_mode, "aggregation": "mean",
                "reward": calculate_docid_reward(
                    {text: keyword_scores[text] for text in texts}, baseline, mode=reward_mode,
                ),
            }, ensure_ascii=False) + "\n")
    (destination / "baseline_keywords.json").write_text(
        json.dumps(
            {"pm": sorted(baseline.pm), "sm": sorted(baseline.sm)},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    summary = {
        "query": query, "baseline_docids": len(ids),
        "reward_mode": reward_mode, "docid_aggregation": "mean",
        "mapped_docids": len(mapping), "id_keyword_pairs": sum(map(len, mapping.values())),
        "distinct_scored_keywords": len(keywords),
        "baseline_pm_keyword_count": len(baseline.pm),
        "baseline_sm_keyword_count": len(baseline.sm),
        "baseline_keyword_count": len(baseline_union),
        "mapping_version": mapping_version, "pm_threshold": PM_THRESHOLD, "sm_threshold": SM_THRESHOLD,
        "pm_meeting_threshold": sum(pm >= PM_THRESHOLD for pm, _ in keyword_scores.values()),
        "sm_meeting_threshold": sum(sm >= SM_THRESHOLD for _, sm in keyword_scores.values()),
        "unique_positive_reward_keywords": sum(row["unique_eligible"] and row["reward"] > 0 for row in rows),
        "top_baseline": sorted((row for row in rows if row["in_baseline_set"]),
                               key=lambda row: row["reward"], reverse=True)[:5],
        "top_outside_baseline": sorted((row for row in rows if not row["in_baseline_set"]),
                                       key=lambda row: row["reward"], reverse=True)[:5],
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    logger.info(
        "Baseline sets ready: %d PM keywords, %d SM keywords",
        len(baseline.pm),
        len(baseline.sm),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--mapping-url", required=True)
    parser.add_argument("--relevance-source")
    parser.add_argument("--relevance-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-queries", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="discrete_relevance_unique")
    args = parser.parse_args()
    if args.max_queries <= 0 or args.batch_size < 4 or args.batch_size % 4:
        raise ValueError("max-queries must be positive and batch-size a positive multiple of 4")
    evidence = json.loads(Path(args.baseline_results).read_text(encoding="utf-8"))
    if evidence.get("infer_mode") != 2 or evidence.get("constraint_function") != "constraint_restrict_decode_vocab":
        raise ValueError("Baseline evidence must use the SLMGR constrained decoder")
    mapping_version = Path(evidence["mapping"]).name
    records = evidence["queries"][:args.max_queries]
    if not records:
        raise ValueError("Baseline evidence contains no queries")
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Baseline output already exists: {output}")
    relevance_source = resolve_relevance_source(args.relevance_model, args.relevance_source)
    logging.basicConfig(level=logging.INFO)
    pipeline_factory = load_relevance_pipeline_factory(relevance_source)
    pipeline = pipeline_factory(
        str(require_directory(args.relevance_model, "Relevance checkpoint")),
        cup="large", device=args.device, dtype="bf16", batch_size=args.batch_size,
        cross_batch_size=256, show_progress=True,
    )
    summaries = [
        _build_query(record, pipeline, args.mapping_url, mapping_version,
                     output / f"query-{index:04d}", args.reward_mode)
        for index, record in enumerate(records)
    ]
    (output / "manifest.json").write_text(json.dumps({
        "baseline_results": str(Path(args.baseline_results).resolve()), "mapping_version": mapping_version,
        "relevance_source": str(relevance_source),
        "relevance_model": str(Path(args.relevance_model).resolve()),
        "queries": summaries, "status": "complete",
        "reward_mode": args.reward_mode, "docid_aggregation": "mean",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

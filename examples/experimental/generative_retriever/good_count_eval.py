"""Run resumable baseline or policy good-count evaluation through two HTTP services."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import httpx
import numpy as np

from examples.experimental.generative_retriever.relevance_reward import PM_THRESHOLD, SM_THRESHOLD


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _index_rows(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    indexed = {}
    for row in rows:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Invalid query in {source}")
        if query in indexed:
            raise ValueError(f"Duplicate query {query!r} in {source}")
        indexed[query] = row
    return indexed


def _generate(client: httpx.Client, url: str, queries: list[str]) -> list[dict[str, Any]]:
    response = client.post(url, json={"queries": queries})
    response.raise_for_status()
    results = response.json().get("results")
    if not isinstance(results, list) or len(results) != len(queries):
        raise ValueError("Policy generation service returned an invalid result count")
    return results


def _score(
    client: httpx.Client,
    url: str,
    queries: list[str],
    generated: list[dict[str, Any]],
    *,
    docids_per_query: int = 64,
) -> list[dict[str, Any]]:
    samples = []
    for query, result in zip(queries, generated, strict=True):
        if result.get("query") != query:
            raise ValueError("Policy generation service changed query ordering")
        rows = result.get("docids")
        if not isinstance(rows, list) or len(rows) != docids_per_query:
            raise ValueError(f"Policy generation service must return {docids_per_query} doc IDs")
        samples.append({
            "query": query,
            "docids": [row["positional_docid"] for row in rows],
        })
    response = client.post(url, json={"samples": samples})
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Good-count service must return a JSON object")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(samples):
        raise ValueError("Good-count service returned an invalid result count")
    version, model = payload.get("mapping_version"), payload.get("relevance_model")
    if not isinstance(version, str) or not version or not isinstance(model, str) or not model:
        raise ValueError("Good-count service returned invalid scoring provenance")
    for query, result in zip(queries, results, strict=True):
        if not isinstance(result, dict) or result.get("query") != query:
            raise ValueError("Good-count service changed query ordering")
        scores = result.get("keyword_scores")
        if not isinstance(scores, list) or not scores or result.get("keyword_count") != len(scores):
            raise ValueError("Good-count service returned invalid keyword scores")
        by_keyword = {}
        for row in scores:
            if not isinstance(row, dict):
                raise ValueError("Good-count service returned an invalid keyword score")
            keyword = row.get("keyword")
            pm, sm = row.get("pm_score"), row.get("sm_score")
            if (
                not isinstance(keyword, str) or not keyword.strip() or keyword in by_keyword
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1
                    for value in (pm, sm)
                )
            ):
                raise ValueError("Good-count service returned duplicate or invalid keyword scores")
            by_keyword[keyword] = (pm, sm)
        for head, column, threshold in (("pm", 0, PM_THRESHOLD), ("sm", 1, SM_THRESHOLD)):
            keywords = result.get(f"{head}_keywords")
            expected = {
                keyword for keyword, scores in by_keyword.items() if scores[column] >= threshold
            }
            if (
                not isinstance(keywords, list)
                or any(not isinstance(keyword, str) for keyword in keywords)
                or len(keywords) != len(expected)
                or set(keywords) != expected
            ):
                raise ValueError(f"Good-count service returned invalid {head.upper()} keywords")
    return [
        {**result, "mapping_version": version, "relevance_model": model}
        for result in results
    ]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("Cannot summarize an empty good-count evaluation")
    counts = np.asarray([row["good_count"] for row in rows], dtype=np.float64)
    median, p10, p90, p95, p99 = np.quantile(counts, [0.50, 0.10, 0.90, 0.95, 0.99])
    summary = {
        "queries": len(rows),
        "good_count_total": int(np.sum(counts)),
        "zero_fraction": float(np.mean(counts == 0)),
        "mean": float(np.mean(counts)),
        "median": float(median),
        "p10": float(p10),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
        "max": int(np.max(counts)),
    }
    for head in ("pm", "sm"):
        values = np.asarray([row[f"{head}_good_count"] for row in rows], dtype=np.float64)
        summary[f"{head}_good_density"] = float(np.mean(values))
        summary[f"{head}_good_rate"] = float(np.mean(values > 0))
    return summary


def run(
    *,
    mode: str,
    manifest: Path,
    output: Path,
    generate_url: str,
    score_url: str,
    baseline_results: Path | None,
    batch_size: int,
    docids_per_query: int = 64,
    score_head: Literal["sm", "pm"] = "sm",
) -> None:
    if mode not in ("baseline", "policy"):
        raise ValueError(f"Unknown evaluation mode: {mode!r}")
    if not 1 <= batch_size <= 512:
        raise ValueError("batch_size must be between 1 and 512")
    if not 1 <= docids_per_query <= 64:
        raise ValueError("docids_per_query must be between 1 and 64")
    if score_head not in ("sm", "pm"):
        raise ValueError(f"Unknown score head: {score_head!r}")
    if mode == "policy" and baseline_results is None:
        raise ValueError("baseline_results is required in policy mode")
    manifest_rows = _read_jsonl(manifest)
    manifest_by_query = _index_rows(manifest_rows, source=manifest)
    if not manifest_rows:
        raise ValueError("Evaluation manifest must not be empty")
    baseline_by_query = (
        _index_rows(_read_jsonl(baseline_results), source=baseline_results)
        if baseline_results is not None
        else {}
    )
    if mode == "policy":
        if any(
            any(field not in row for field in (
                "baseline_pm_keywords", "baseline_sm_keywords",
                "keyword_scores", "mapping_version", "relevance_model",
            ))
            for row in baseline_by_query.values()
        ):
            raise ValueError("Baseline results lack PM/SM scores; rerun baseline evaluation")
        missing = manifest_by_query.keys() - baseline_by_query.keys()
        if missing:
            raise ValueError(f"Baseline results missing for {len(missing)} manifest queries")
    processed = _index_rows(_read_jsonl(output), source=output) if output.exists() else {}
    if processed.keys() - manifest_by_query.keys():
        raise ValueError("Existing output contains queries outside the manifest")
    required_fields = (
        ("baseline_pm_keywords", "baseline_sm_keywords", "keyword_scores",
         "mapping_version", "relevance_model")
        if mode == "baseline" else
        ("policy_pm_keywords", "policy_sm_keywords", "pm_good_keywords", "sm_good_keywords",
         "pm_good_count", "sm_good_count", "keyword_scores", "good_keywords",
         "good_count", "mapping_version", "relevance_model")
    )
    if any(any(field not in row for field in required_fields) for row in processed.values()):
        raise ValueError("Existing output does not match the evaluation mode")
    pending = [row for row in manifest_rows if row["query"] not in processed]
    with httpx.Client(timeout=900.0) as client, ThreadPoolExecutor(max_workers=1) as executor:
        if pending:
            generation_future = executor.submit(
                _generate, client, generate_url,
                [row["query"] for row in pending[:batch_size]],
            )
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                queries = [row["query"] for row in batch]
                generated = generation_future.result()
                next_batch = pending[start + batch_size:start + 2 * batch_size]
                if next_batch:
                    generation_future = executor.submit(
                        _generate, client, generate_url, [row["query"] for row in next_batch],
                    )
                scored = _score(
                    client, score_url, queries, generated,
                    docids_per_query=docids_per_query,
                )
                output_rows = []
                for manifest_row, generation, score in zip(batch, generated, scored, strict=True):
                    query = manifest_row["query"]
                    row = {
                        **manifest_row,
                        "docids": [item["positional_docid"] for item in generation["docids"]],
                        "keyword_scores": score["keyword_scores"],
                        "mapping_version": score["mapping_version"],
                        "relevance_model": score["relevance_model"],
                    }
                    if mode == "baseline":
                        for head in ("pm", "sm"):
                            row[f"baseline_{head}_keywords"] = sorted(score[f"{head}_keywords"])
                    else:
                        baseline = baseline_by_query[query]
                        if (
                            baseline["mapping_version"] != score["mapping_version"]
                            or baseline["relevance_model"] != score["relevance_model"]
                        ):
                            raise ValueError(f"Baseline uses different scoring model for query {query!r}")
                        for head in ("pm", "sm"):
                            qualified = frozenset(score[f"{head}_keywords"])
                            good = sorted(qualified - frozenset(baseline[f"baseline_{head}_keywords"]))
                            row[f"policy_{head}_keywords"] = sorted(qualified)
                            row[f"{head}_good_keywords"] = good
                            row[f"{head}_good_count"] = len(good)
                        row["good_keywords"] = row[f"{score_head}_good_keywords"]
                        row["good_count"] = row[f"{score_head}_good_count"]
                    output_rows.append(row)
                _write_rows(output, output_rows)
                print(json.dumps({"completed": len(processed) + start + len(output_rows), "total": len(manifest_rows)}))
    if mode == "policy":
        print(json.dumps(_summary(output), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", "policy"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate-url", required=True)
    parser.add_argument("--score-url", required=True)
    parser.add_argument("--baseline-results", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--docids-per-query", type=int, default=64)
    parser.add_argument("--score-head", choices=("sm", "pm"), default="sm")
    args = parser.parse_args()
    run(
        mode=args.mode,
        manifest=args.manifest,
        output=args.output,
        generate_url=args.generate_url,
        score_url=args.score_url,
        baseline_results=args.baseline_results,
        batch_size=args.batch_size,
        docids_per_query=args.docids_per_query,
        score_head=args.score_head,
    )


if __name__ == "__main__":
    main()

"""Compute evaluation metrics and produce a summary report.

Extracted from the aggregation logic in the original ``judge.py``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)


@register_resource("metrics_summary")
class MetricsSummaryResource(BaseResource):
    """Aggregate graded results into accuracy metrics and optional JSON report."""

    def __init__(
        self,
        output_dir: str = "output",
        task_name: str = "eval",
        **_kwargs: Any,
    ) -> None:
        self.output_dir = output_dir
        self.task_name = task_name

    async def process(self, samples: dict) -> tuple:
        """Aggregate all sample grades into a summary.

        *samples* is the ``benchmark.samples`` dict produced by the pipeline.
        Returns ``(summary_dict,)``.
        """
        all_grades: list[dict] = []
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for _key, sample in sorted(samples.items()):
            # Per-question grades from streaming pipeline (grade.N traces)
            grade_dict = sample.get("grade")
            if isinstance(grade_dict, dict):
                all_grades.extend(grade_dict.values())
            # Legacy: batch grades from old pipeline
            elif isinstance(sample.get("grades"), list):
                all_grades.extend(sample["grades"])

            # Usage from per-question items
            qa_items = sample.get("qa_item")
            if isinstance(qa_items, dict):
                for r in qa_items.values():
                    usage = r.get("usage", {})
                    for k in total_usage:
                        total_usage[k] += usage.get(k, 0)
            # Legacy fallback
            elif isinstance(sample.get("qa_results"), list):
                for r in sample["qa_results"]:
                    usage = r.get("usage", {})
                    for k in total_usage:
                        total_usage[k] += usage.get(k, 0)

        total = len(all_grades)
        correct = sum(1 for g in all_grades if g.get("grade"))
        score = correct / total if total > 0 else 0.0

        # Per-category breakdown
        categories: dict[str, dict[str, int]] = {}
        for g in all_grades:
            cat = str(g.get("category", "unknown"))
            categories.setdefault(cat, {"correct": 0, "total": 0})
            categories[cat]["total"] += 1
            if g.get("grade"):
                categories[cat]["correct"] += 1

        per_category = {}
        for cat in sorted(categories):
            c = categories[cat]
            pct = c["correct"] / c["total"] if c["total"] > 0 else 0.0
            per_category[cat] = {**c, "accuracy": round(pct, 4)}

        summary = {
            "score": round(score, 4),
            "correct": correct,
            "total": total,
            "per_category": per_category,
            "total_usage": total_usage,
        }

        # Log results
        logger.info("=" * 50)
        logger.info("Results: %d/%d correct (%.2f%%)", correct, total, score * 100)
        if len(per_category) > 1:
            for cat, info in per_category.items():
                logger.info(
                    "  Category %s: %d/%d (%.2f%%)",
                    cat, info["correct"], info["total"], info["accuracy"] * 100,
                )
        logger.info("Total tokens: %s", total_usage)
        logger.info("=" * 50)

        # Write output files
        os.makedirs(self.output_dir, exist_ok=True)

        grades_path = os.path.join(self.output_dir, f"grades.{self.task_name}.json")
        with open(grades_path, "w", encoding="utf-8") as f:
            json.dump(
                {"score": score, "correct": correct, "total": total, "grades": all_grades},
                f, indent=2, ensure_ascii=False,
            )
        logger.info("Grades written to %s", grades_path)

        summary_path = os.path.join(self.output_dir, f"summary.{self.task_name}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info("Summary written to %s", summary_path)

        return (summary,)

"""Load pre-existing QA results from a JSONL file.

Used by the judge-only pipeline where QA has already been performed and
results are persisted on disk.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)


@register_resource("qa_results_reader")
class QAResultsReaderResource(BaseResource):
    """Read a QA JSONL/JSON file and group records by sample_id."""

    async def process(self, path: str) -> tuple:
        """Returns ``(samples_dict,)`` keyed by sample index string."""
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                records = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    records = data.get("results", data.get("grades", []))
                elif isinstance(data, list):
                    records = data

        # Group by sample_id (or treat all as sample "0")
        grouped: dict[str, list[dict]] = {}
        for r in records:
            key = str(r.get("sample_id", "0"))
            grouped.setdefault(key, []).append(r)

        samples: dict[str, dict] = {}
        for idx, (sample_id, recs) in enumerate(sorted(grouped.items())):
            samples[str(idx)] = {
                "sample_id": sample_id,
                "qa_results": recs,
            }

        logger.info(
            "Loaded %d QA records across %d samples from %s",
            len(records), len(samples), path,
        )
        return (samples,)

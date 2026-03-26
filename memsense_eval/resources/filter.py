"""Filter out bad / error QA responses before judging.

Extracted from the original ``filter_jsonl.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)

_ERROR_PATTERNS = (
    "LLM request timed out.",
    "returned a billing error",
    "[ERROR] HTTPConnectionPool(host='127.0.0.1'",
    "The AI service is temporarily overloaded.",
    "Unauthorized - Invalid token",
    "503 no healthy upstream",
    "无可用渠道",
)


@register_resource("response_filter")
class ResponseFilterResource(BaseResource):
    """Drop QA records whose response matches known error patterns."""

    def __init__(self, extra_patterns: list[str] | None = None, **_kwargs: Any) -> None:
        self.patterns = list(_ERROR_PATTERNS)
        if extra_patterns:
            self.patterns.extend(extra_patterns)

    async def process(self, qa_results: list[dict]) -> tuple:
        filtered = [
            r for r in qa_results
            if not any(pat in r.get("response", "") for pat in self.patterns)
        ]
        dropped = len(qa_results) - len(filtered)
        if dropped:
            logger.info("Filtered out %d bad responses", dropped)
        return (filtered,)

"""Shared evaluation state addressed by *trace* paths.

A trace is a dot-separated path into a nested dict, e.g.
``benchmark.samples.0.grades``.  The DataManager provides async-safe
read / write and enumerates all "ready" (populated) traces so the
pipeline engine can decide which flows are runnable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DataManager:
    """Thread-safe nested-dict store with trace-path addressing."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self._lock = asyncio.Lock()

    # ---- read (lock-free, dict is only appended to) ----

    def get_trace_data(self, trace: list[str]) -> Any:
        cur = self.data
        for seg in trace:
            cur = cur[seg]
        return cur

    def get_all_ready_traces(self) -> list[list[str]]:
        traces: list[list[str]] = []

        def _walk(node: Any, prefix: list[str]) -> None:
            if not isinstance(node, dict):
                return
            for key, val in node.items():
                path = [*prefix, key]
                traces.append(path)
                _walk(val, path)

        _walk(self.data, [])
        return traces

    # ---- write (async-locked) ----

    async def set_trace_data(self, trace: list[str], value: Any) -> None:
        async with self._lock:
            cur = self.data
            for seg in trace[:-1]:
                if seg not in cur:
                    cur[seg] = {}
                cur = cur[seg]
            existing = cur.get(trace[-1])
            if existing is not None:
                logger.warning("Overwriting trace %s", ".".join(trace))
            cur[trace[-1]] = value

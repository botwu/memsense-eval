"""Pipeline engine — simplified, auto-terminating FlowSquare.

The engine polls ready traces each tick, matches them against declared
flows, and dispatches resource execution as async tasks.  Unlike the
original ef FlowSquare it will **automatically stop** once no new work
can be produced for two consecutive idle cycles, making it suitable for
batch CLI evaluation runs.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from pydantic import BaseModel

from memsense_eval.engine.data_manager import DataManager
from memsense_eval.engine.flow import (
    FlowConfig,
    apply_mapping_to_traces,
    match_traces,
)
from memsense_eval.engine.resource import ResourceConfig, create_resource

logger = logging.getLogger(__name__)


class PipelineConfig(BaseModel):
    data: dict[str, Any]
    flows: list[FlowConfig]
    resources: list[ResourceConfig]


class _TaskInfo:
    """Bookkeeping wrapper around a dispatched coroutine."""

    def __init__(self, coro, flow_config: FlowConfig, inputs: list[Any]):
        self.coro = coro
        self.flow_config = flow_config
        self.inputs = inputs


class PipelineEngine:
    """Async evaluation pipeline driven by flow declarations."""

    def __init__(self, config: PipelineConfig) -> None:
        self.flow_configs = config.flows
        self.resources: dict[str, Any] = {}
        for rc in config.resources:
            self.resources[rc.name] = create_resource(rc)
            logger.info("Created resource: %s", rc.name)

        self.data_manager = DataManager(config.data)

        # Track which traces have already been consumed per flow
        self.assigned: dict[FlowConfig, set[str]] = {
            fc: set() for fc in self.flow_configs
        }
        # For reuse flows: snapshot of the input data fingerprint at last
        # dispatch, so we only re-run when upstream data has changed.
        self._reuse_last_fingerprint: dict[FlowConfig, str] = {}
        self._stop = False
        self._pending_tasks: set[asyncio.Task] = set()

    # ---- public API ----

    async def start(self, tick_interval: float = 1.0, idle_limit: int = 3) -> None:
        """Run the pipeline until completion or manual stop.

        *idle_limit* consecutive ticks without new work triggers auto-stop.
        """
        idle_count = 0
        while not self._stop:
            dispatched = await self._run_once()
            if dispatched == 0 and not self._pending_tasks:
                idle_count += 1
                if idle_count >= idle_limit:
                    logger.info(
                        "No new work for %d ticks and no pending tasks — stopping.",
                        idle_limit,
                    )
                    break
            else:
                idle_count = 0
            await asyncio.sleep(tick_interval)

        # Wait for stragglers
        if self._pending_tasks:
            logger.info("Waiting for %d remaining tasks …", len(self._pending_tasks))
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)

    def stop(self) -> None:
        self._stop = True

    # ---- internals ----

    async def _run_once(self) -> int:
        """Single scheduling tick.  Returns number of newly dispatched tasks."""
        ready = self.data_manager.get_all_ready_traces()
        ready_set = {".".join(t) for t in ready}
        ready_count = len(ready_set)
        logger.debug("Ready traces: %d", ready_count)

        dispatched = 0
        for fc in self.flow_configs:
            if fc.by is None:
                continue

            # For reuse flows, only re-run if the data they read has changed.
            if fc.reuse:
                try:
                    inputs = [self.data_manager.get_trace_data(t) for t in fc.by]
                    fingerprint = repr(inputs)
                except (KeyError, TypeError):
                    fingerprint = ""
                if fingerprint == self._reuse_last_fingerprint.get(fc):
                    continue

            while True:
                possible_set = ready_set - self.assigned[fc]
                possible = [s.split(".") for s in possible_set]

                matched, mapping, by_traces = match_traces(fc.by, possible)
                if not matched or mapping is None:
                    break

                obtain_traces = (
                    apply_mapping_to_traces(mapping, fc.obtain)
                    if fc.obtain
                    else []
                )

                by_info = " | ".join(".".join(t) for t in by_traces)
                obtain_info = " | ".join(".".join(t) for t in obtain_traces)
                logger.info("Dispatch [%s]: %s → %s", fc.use, by_info, obtain_info)

                coro = self._run_flow(fc, by_traces, obtain_traces)
                task = asyncio.create_task(self._track(coro, fc, by_traces))
                self._pending_tasks.add(task)
                task.add_done_callback(self._pending_tasks.discard)
                dispatched += 1

                if not fc.reuse:
                    for t in by_traces:
                        self.assigned[fc].add(".".join(t))
                else:
                    try:
                        inputs = [self.data_manager.get_trace_data(t) for t in by_traces]
                        self._reuse_last_fingerprint[fc] = repr(inputs)
                    except (KeyError, TypeError):
                        pass
                    break

        return dispatched

    async def _run_flow(
        self,
        fc: FlowConfig,
        by_traces: list[list[str]],
        obtain_traces: list[list[str]],
    ) -> None:
        resource = self.resources.get(fc.use)
        if resource is None:
            raise ValueError(f"Resource '{fc.use}' not found")

        inputs = [self.data_manager.get_trace_data(t) for t in by_traces]
        output = await resource.process(*inputs)

        if not isinstance(output, tuple):
            raise TypeError(
                f"Resource '{fc.use}' must return a tuple, got {type(output).__name__}"
            )
        if len(output) != len(obtain_traces):
            raise ValueError(
                f"Resource '{fc.use}' returned {len(output)} values "
                f"but {len(obtain_traces)} obtain traces declared"
            )

        for trace, value in zip(obtain_traces, output):
            await self.data_manager.set_trace_data(trace, value)

    async def _track(
        self,
        coro,
        fc: FlowConfig,
        by_traces: list[list[str]],
    ) -> None:
        try:
            await coro
        except Exception:
            by_info = " | ".join(".".join(t) for t in by_traces)
            logger.error(
                "Flow [%s] failed for %s:\n%s",
                fc.use,
                by_info,
                traceback.format_exc(),
            )

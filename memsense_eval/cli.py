"""Unified CLI entry point for the Memsense evaluation framework.

Supports two modes:

* **CLI mode** (default) — run the pipeline to completion, then exit.
* **Server mode** (``--serve``) — start a FastAPI server that exposes
  ``/data`` (live state) and ``/stop`` while the pipeline runs in the
  background.

Environment variable interpolation is supported in YAML values via the
``${VAR}`` syntax, and individual data keys can be overridden from the
command line with ``--set key=value``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from typing import Any

import yaml

# Ensure resource plugins are registered before anything else
import memsense_eval.resources  # noqa: F401

from memsense_eval.engine.flow import FlowConfig
from memsense_eval.engine.pipeline import PipelineConfig, PipelineEngine
from memsense_eval.engine.resource import ResourceConfig

logger = logging.getLogger("memsense_eval")

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _resolve_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` with environment variable values.

    If the **entire** string is a single ``${VAR}`` and the variable is not
    set, the value becomes ``None`` so downstream code can fall back to its
    own defaults.  Inline ``${VAR}`` in a larger string is left as-is when
    the variable is missing (for easier debugging).
    """
    if isinstance(value, str):
        full = _ENV_RE.fullmatch(value)
        if full:
            return os.environ.get(full.group(1))
        def _repl(m: re.Match) -> str:
            return os.environ.get(m.group(1), m.group(0))
        return _ENV_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _apply_overrides(data: dict, overrides: list[str]) -> None:
    """Apply ``--set key.path=value`` overrides to the data dict."""
    for item in overrides:
        if "=" not in item:
            logger.warning("Ignoring malformed --set: %s", item)
            continue
        key, val = item.split("=", 1)
        parts = key.split(".")
        cur = data
        for seg in parts[:-1]:
            cur = cur.setdefault(seg, {})
        cur[parts[-1]] = val


def load_config(path: str, overrides: list[str] | None = None) -> PipelineConfig:
    """Parse a YAML config file into a :class:`PipelineConfig`."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _resolve_env(raw)

    data = raw.get("data", {})
    if overrides:
        _apply_overrides(data, overrides)

    flows = [FlowConfig.from_dict(f) for f in raw.get("flows", [])]
    resources = [ResourceConfig(**r) for r in raw.get("resources", [])]

    return PipelineConfig(data=data, flows=flows, resources=resources)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memsense_eval",
        description="Memsense Evaluation Framework — run YAML-driven pipelines",
    )
    p.add_argument("config", help="Path to a YAML pipeline config file")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override data keys: --set benchmark.token=xxx",
    )
    p.add_argument(
        "--tick",
        type=float,
        default=1.0,
        help="Pipeline tick interval in seconds (default: 1.0)",
    )
    p.add_argument(
        "--idle-limit",
        type=int,
        default=3,
        help="Auto-stop after N idle ticks (default: 3)",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Run as FastAPI server with /data and /stop endpoints",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Server port when --serve is used (default: 8003)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p


async def run_cli(config: PipelineConfig, tick: float, idle_limit: int) -> None:
    engine = PipelineEngine(config)
    await engine.start(tick_interval=tick, idle_limit=idle_limit)
    logger.info("Pipeline finished.")


def run_server(config: PipelineConfig, port: int, tick: float) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI
    except ImportError:
        logger.error("Server mode requires 'fastapi' and 'uvicorn'. Install them first.")
        sys.exit(1)

    app = FastAPI(title="Memsense Eval API")
    engine = PipelineEngine(config)
    bg_task: asyncio.Task | None = None

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal bg_task
        bg_task = asyncio.create_task(engine.start(tick_interval=tick))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine.stop()
        if bg_task:
            await bg_task

    @app.get("/data")
    async def _data():
        return engine.data_manager.data

    @app.get("/stop")
    async def _stop():
        engine.stop()
        return {"msg": "stopping"}

    uvicorn.run(app, host="0.0.0.0", port=port)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)

    config = load_config(args.config, args.overrides)

    if args.serve:
        run_server(config, args.port, args.tick)
    else:
        asyncio.run(run_cli(config, args.tick, args.idle_limit))

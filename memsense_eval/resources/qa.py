"""Run QA questions against OpenClaw agent (via CLI).

The OpenClaw gateway exposes a WebSocket interface, not a REST
``/v1/responses`` endpoint.  This resource calls ``openclaw agent``
CLI with ``--json`` to send each question and parse the response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
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


def _reset_session_file(session_id: str) -> None:
    """Archive the session JSONL to prevent history accumulation."""
    sessions_dir = os.path.expanduser("~/.openclaw/agents/main/sessions")
    src = os.path.join(sessions_dir, f"{session_id}.jsonl")
    if not os.path.exists(src):
        return
    dst = f"{src}.reset.{time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())}Z"
    try:
        os.rename(src, dst)
        logger.debug("Archived session %s", session_id)
    except Exception as exc:
        logger.warning("Session reset error: %s", exc)


def _send_via_cli(
    message: str,
    session_id: str,
    agent_id: str = "main",
    timeout: int = 300,
) -> tuple[str, dict]:
    """Call ``openclaw agent`` CLI and return (response_text, usage_dict)."""
    cmd = [
        "openclaw", "agent",
        "--agent", agent_id,
        "--session-id", session_id,
        "--message", message,
        "--json",
        "--timeout", str(timeout),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout + 30,
    )

    if result.returncode != 0:
        stderr_snippet = (result.stderr or "")[-300:]
        raise RuntimeError(
            f"openclaw agent exited {result.returncode}: {stderr_snippet}"
        )

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("openclaw agent returned empty output")

    # stdout may contain plugin log lines before the JSON — find the JSON
    json_start = stdout.find("{")
    if json_start < 0:
        raise RuntimeError(f"No JSON in openclaw output: {stdout[:200]}")

    body = json.loads(stdout[json_start:])

    if body.get("status") != "ok":
        raise RuntimeError(f"openclaw agent error: {body.get('summary', body)}")

    payloads = body.get("result", {}).get("payloads", [])
    response_text = payloads[0].get("text", "") if payloads else ""

    agent_meta = body.get("result", {}).get("meta", {}).get("agentMeta", {})
    raw_usage = agent_meta.get("usage", {})
    usage = {
        "input_tokens": raw_usage.get("input", 0),
        "output_tokens": raw_usage.get("output", 0),
        "total_tokens": raw_usage.get("total", 0),
    }

    return response_text, usage


def _is_error_response(text: str) -> bool:
    return any(pat in text for pat in _ERROR_PATTERNS)


@register_resource("memsense_qa")
class MemsenseQAResource(BaseResource):
    """Send QA questions to OpenClaw agent and collect answers."""

    def __init__(
        self,
        agent_id: str = "main",
        concurrency: int = 1,
        retries: int = 2,
        timeout: int = 300,
        user_prefix: str = "eval",
        # Legacy HTTP params kept for config compat — ignored in CLI mode
        base_url: str | None = None,
        token: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self.agent_id = agent_id
        self.concurrency = concurrency
        self.retries = retries
        self.timeout = timeout
        self.user_prefix = user_prefix

    async def process(self, sample: dict, _ingest_result: Any = None) -> tuple:
        """Run QA for a single sample.

        Each question uses a unique session ID so the memsense plugin
        performs a fresh memory search per question.
        Returns ``(qa_results_list,)``.
        """
        sample_id = sample["sample_id"]
        qa_list: list[dict] = sample["qa_list"]

        sem = asyncio.Semaphore(self.concurrency)

        async def _ask(qi: int, qa: dict) -> dict | None:
            async with sem:
                question = qa["question"]
                expected = str(qa["answer"])
                category = qa.get("category", "")
                evidence = qa.get("evidence", [])

                session_id = f"{self.user_prefix}-{sample_id}-q{qi}"
                logger.info("[%s] Q%d: %s", sample_id, qi, question[:60])

                for attempt in range(self.retries):
                    try:
                        response, usage = await asyncio.to_thread(
                            _send_via_cli,
                            question,
                            session_id,
                            self.agent_id,
                            self.timeout,
                        )

                        if _is_error_response(response):
                            if attempt < self.retries - 1:
                                logger.warning(
                                    "[%s] Q%d error response, retrying…",
                                    sample_id, qi,
                                )
                                await asyncio.sleep(1.0)
                                continue
                            logger.warning("[%s] Q%d max retries reached", sample_id, qi)
                            return None

                        logger.info("[%s] Q%d → %s", sample_id, qi, response[:80])

                        _reset_session_file(session_id)

                        return {
                            "sample_id": sample_id,
                            "qi": qi,
                            "question": question,
                            "expected": expected,
                            "response": response,
                            "category": category,
                            "evidence": evidence,
                            "usage": usage,
                        }

                    except Exception as exc:
                        if attempt < self.retries - 1:
                            logger.warning(
                                "[%s] Q%d exception: %s, retrying…",
                                sample_id, qi, exc,
                            )
                            await asyncio.sleep(1.0)
                        else:
                            logger.error("[%s] Q%d failed: %s", sample_id, qi, exc)
                            return None
                return None

        tasks = [_ask(qi + 1, qa) for qi, qa in enumerate(qa_list)]
        results = await asyncio.gather(*tasks)
        records = [r for r in results if r is not None]

        total_usage = {
            "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in records),
            "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in records),
            "total_tokens": sum(r["usage"].get("total_tokens", 0) for r in records),
        }
        logger.info(
            "[%s] QA complete: %d/%d answered, tokens=%s",
            sample_id, len(records), len(qa_list), total_usage,
        )

        return (records,)

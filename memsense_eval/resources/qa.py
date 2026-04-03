"""Run QA questions against Memsense/OpenClaw with dual-mode support.

Supports both HTTP and CLI modes:
- HTTP mode: POST to /v1/responses (if service exists)
- CLI mode: Call openclaw agent via subprocess (fallback)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from typing import Any, Literal

import aiohttp

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


def _is_error_response(text: str) -> bool:
    return any(pat in text for pat in _ERROR_PATTERNS)


# ============================================================================
# HTTP Mode Implementation
# ============================================================================

async def _send_message_http(
    session: aiohttp.ClientSession,
    base_url: str,
    token: str,
    user: str,
    message: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
    timeout: int = 300,
) -> tuple[str, dict]:
    """Send message via HTTP POST to /v1/responses."""
    url = f"{base_url}/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    payload = {
        "model": "openclaw",
        "input": message,
        "stream": False,
        "user": user,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if not resp.ok:
                    error_text = await resp.text()
                    raise RuntimeError(f"API error {resp.status}: {error_text[:200]}")

                body = await resp.json()

                # Extract response text
                response_text = ""
                for item in body.get("output", []):
                    if item.get("type") == "message":
                        for content in item.get("content", []):
                            if content.get("type") == "output_text":
                                response_text = content.get("text", "")
                                break

                usage = body.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

                if _is_error_response(response_text):
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), 10.0)
                        logger.warning(
                            "Error pattern detected (attempt %d/%d), retrying in %.1fs",
                            attempt + 1, max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.warning("Error pattern detected after max retries")
                    return response_text, usage

                return response_text, usage

        except Exception as exc:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 10.0)
                logger.warning(
                    "HTTP request failed (attempt %d/%d): %s, retrying in %.1fs",
                    attempt + 1, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("HTTP request failed after %d attempts: %s", max_retries, exc)
                raise

    raise RuntimeError(f"Failed after {max_retries} attempts")


# ============================================================================
# CLI Mode Implementation
# ============================================================================

async def _send_message_cli(
    message: str,
    session_id: str,
    agent_id: str = "main",
    timeout: int = 300,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> tuple[str, dict]:
    """Send message via openclaw agent CLI."""
    cmd = [
        "openclaw", "agent",
        "--agent", agent_id,
        "--session-id", session_id,
        "--message", message,
        "--json",
        "--timeout", str(timeout),
    ]

    for attempt in range(max_retries):
        try:
            result = await asyncio.to_thread(
                subprocess.run,
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

            # Find JSON in output
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

            if _is_error_response(response_text):
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), 10.0)
                    logger.warning(
                        "Error pattern detected (attempt %d/%d), retrying in %.1fs",
                        attempt + 1, max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("Error pattern detected after max retries")
                return response_text, usage

            return response_text, usage

        except Exception as exc:
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), 10.0)
                logger.warning(
                    "CLI request failed (attempt %d/%d): %s, retrying in %.1fs",
                    attempt + 1, max_retries, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("CLI request failed after %d attempts: %s", max_retries, exc)
                raise

    raise RuntimeError(f"Failed after {max_retries} attempts")


# ============================================================================
# Session Management (file-based, matching original evaluation)
# ============================================================================

def _load_existing_answers(jsonl_path: str) -> tuple[list[dict], set[str]]:
    """Load existing QA answers from JSONL file for resume support."""
    records: list[dict] = []
    questions: set[str] = set()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    records.append(rec)
                    questions.add(rec["question"])
    except FileNotFoundError:
        pass
    return records, questions


async def _get_session_id(user_key: str) -> str | None:
    """Read the current session ID for the given user from sessions.json."""
    sessions_file = os.path.expanduser("~/.openclaw/agents/main/sessions/sessions.json")
    try:
        data = await asyncio.to_thread(_read_json, sessions_file)
        key = f"agent:main:openresponses-user:{user_key}"
        return data.get(key, {}).get("sessionId")
    except Exception as exc:
        logger.debug("Could not read session ID for %s: %s", user_key, exc)
        return None


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


async def _reset_session(user_key: str) -> None:
    """Archive openclaw agent session .jsonl to prevent history accumulation.

    Mirrors the original evaluation/qa.py ``reset_session`` behaviour:
    rename ``<session_id>.jsonl`` → ``<session_id>.jsonl.<epoch>``.
    """
    session_id = await _get_session_id(user_key)
    if not session_id:
        return

    sessions_dir = os.path.expanduser("~/.openclaw/agents/main/sessions")
    src = os.path.join(sessions_dir, f"{session_id}.jsonl")
    if not os.path.exists(src):
        return

    dst = f"{src}.{int(time.time())}"
    try:
        await asyncio.to_thread(os.rename, src, dst)
        logger.debug("Archived session %s", session_id)
    except Exception as exc:
        logger.warning("Session reset failed for %s: %s", user_key, exc)


# ============================================================================
# Resource Implementation
# ============================================================================

@register_resource("memsense_qa")
class MemsenseQAResource(BaseResource):
    """Send QA questions with dual-mode support (HTTP or CLI).

    Questions within a sample are processed **concurrently**. Each question
    uses a unique per-question user key (``{prefix}-{sample}-q{N}``) so that
    openclaw assigns independent sessions — no session reset needed.

    The ``memsense_test_`` prefix in user_prefix causes the plugin to skip
    auto-capture, preventing QA answers from polluting the memory store.
    The memsense plugin strips the ``-q{N}`` suffix before memory search,
    so all questions retrieve from the correct base user's memories.
    """

    def __init__(
        self,
        mode: Literal["http", "cli"] = "http",
        # HTTP mode params
        base_url: str = "http://127.0.0.1:8899",
        token: str | None = None,
        # CLI mode params
        agent_id: str = "main",
        # Common params
        max_retries: int = 3,
        base_delay: float = 2.0,
        timeout: int = 300,
        user_prefix: str = "eval",
        output_dir: str = "output",
        **_kwargs: Any,
    ) -> None:
        self.mode = mode
        self.base_url = base_url
        self.token = token
        self.agent_id = agent_id
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.timeout = timeout
        self.user_prefix = user_prefix
        self.output_dir = output_dir

        if mode == "http" and not token:
            raise ValueError("HTTP mode requires 'token' parameter")

        logger.info("QA resource initialized in %s mode", mode.upper())

    async def process(self, sample: dict, _embeddings_ready: Any = None) -> tuple:
        """Run QA for a single sample with **concurrent** question processing.

        Each question gets a unique user key for session isolation. Memory
        search uses ``MEMSENSE_EVAL_USER_ID`` so all questions find the same
        memories. No session reset is needed.
        """
        sample_id = sample["sample_id"]
        qa_list: list[dict] = sample["qa_list"]
        user_key_base = f"{self.user_prefix}-{sample_id}"

        # Resume support: load existing answers
        os.makedirs(self.output_dir, exist_ok=True)
        jsonl_path = os.path.join(self.output_dir, f"qa.{sample_id}.jsonl")
        existing_records, existing_questions = _load_existing_answers(jsonl_path)
        if existing_questions:
            logger.info(
                "[%s] Resuming — %d questions already answered, skipping",
                sample_id, len(existing_questions),
            )

        records: list[dict] = list(existing_records)
        lock = asyncio.Lock()  # Protect JSONL writes in concurrent context

        # Filter questions to new ones
        new_questions = [
            (qi, qa) for qi, qa in enumerate(qa_list, start=1)
            if qa["question"] not in existing_questions
        ]

        if not new_questions:
            logger.info("[%s] All questions already answered", sample_id)
            return (records,)

        logger.info("[%s] Processing %d new questions with concurrency=5", sample_id, len(new_questions))

        # Pre-create qa_item dict on the sample for streaming dispatch.
        # As each question completes, its record is written here so the
        # pipeline engine can discover it on the next tick and immediately
        # dispatch downstream flows (e.g. per-question judging).
        sample.setdefault("qa_item", {})

        sem = asyncio.Semaphore(5)  # Max 5 concurrent questions

        async def process_one_question(qi: int, qa: dict) -> dict | None:
            """Process a single question with concurrent semaphore control."""
            async with sem:
                question = qa["question"]
                expected = str(qa["answer"])
                category = qa.get("category", "")
                evidence = qa.get("evidence", [])

                logger.info("[%s] Q%d/%d: %s", sample_id, qi, len(qa_list), question[:60])

                try:
                    question_user = f"{user_key_base}-q{qi}"
                    if self.mode == "http":
                        async with aiohttp.ClientSession(trust_env=False) as session:
                            response, usage = await _send_message_http(
                                session,
                                self.base_url,
                                self.token,
                                question_user,
                                question,
                                self.max_retries,
                                self.base_delay,
                                self.timeout,
                            )
                    else:  # CLI mode
                        response, usage = await _send_message_cli(
                            question,
                            question_user,
                            self.agent_id,
                            self.timeout,
                            self.max_retries,
                            self.base_delay,
                        )

                    if _is_error_response(response):
                        logger.warning("[%s] Q%d error response after retries", sample_id, qi)
                        return None

                    if usage.get("output_tokens", 0) > 4000:
                        logger.warning(
                            "[%s] Q%d abnormally long response: %d tokens",
                            sample_id, qi, usage["output_tokens"],
                        )

                    logger.info("[%s] Q%d → %s", sample_id, qi, response[:80])

                    record = {
                        "sample_id": sample_id,
                        "qi": qi,
                        "question": question,
                        "expected": expected,
                        "response": response,
                        "category": category,
                        "evidence": evidence,
                        "usage": usage,
                    }

                    # Save immediately to JSONL for resume support (with lock)
                    async with lock:
                        with open(jsonl_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # Emit per-question trace for streaming judge dispatch.
                    # The engine's _walk discovers this on the next tick.
                    sample["qa_item"][str(qi)] = record

                    return record

                except Exception as exc:
                    logger.error("[%s] Q%d failed: %s", sample_id, qi, exc)
                    return None

        # Launch concurrent tasks
        tasks = [process_one_question(qi, qa) for qi, qa in new_questions]
        results = await asyncio.gather(*tasks)

        # Collect results
        for result in results:
            if result is not None:
                records.append(result)

        total_usage = {
            "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in records),
            "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in records),
            "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in records),
        }
        logger.info(
            "[%s] QA complete: %d/%d answered, tokens=%s",
            sample_id, len(records), len(qa_list), total_usage,
        )

        return (records,)

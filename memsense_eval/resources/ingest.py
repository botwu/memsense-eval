"""Ingest conversations into the Memsense memory API.

Async port of ``memsense/evaluation/ingest.py`` with:
- aiohttp with ``trust_env=False`` to bypass system proxy
- Session cleanup after ingest

Note: tagging/facets are handled entirely by the memsense tag-worker
(direct model API), not by this ingest resource.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import aiohttp

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_locomo_datetime(dt_str: str) -> int | None:
    """Parse LoCoMo date like '1:56 pm on 8 May, 2023' into epoch ms."""
    if not dt_str:
        return None
    try:
        return int(datetime.strptime(dt_str, "%I:%M %p on %d %B, %Y").timestamp() * 1000)
    except (ValueError, TypeError):
        return None


def _extract_speaker_lines(text: str, speaker_name: str) -> str:
    """Extract lines belonging to a specific speaker from conversation text."""
    result = []
    capturing = False
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(f"{speaker_name}:"):
            capturing = True
            result.append(stripped)
        elif ":" in stripped and not stripped.startswith("[") and not stripped.startswith("http"):
            capturing = False
        elif capturing:
            result.append(stripped)
        elif stripped.startswith("["):
            result.append(stripped)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Save to memsense API
# ---------------------------------------------------------------------------

async def _save_to_memsense(
    session: aiohttp.ClientSession,
    base_url: str,
    content: str,
    user_key: str,
    session_key: str,
    token: str | None = None,
    max_chunk_size: int = 4000,
    speaker_a_text: str = "",
    speaker_b_text: str = "",
    date_time: str = "",
    speaker_b_name: str = "",
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> list[dict]:
    """Save content to memsense API with chunking and exponential backoff retry."""
    url = f"{base_url}/v1/memory/save"

    # Chunk long content
    if len(content) <= max_chunk_size:
        chunks = [(content, speaker_a_text, speaker_b_text)]
    else:
        parts = content.split("\n\n")
        chunk_list: list[tuple[str, str, str]] = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 <= max_chunk_size:
                current += ("\n\n" if current else "") + part
            else:
                if current:
                    b_chunk = _extract_speaker_lines(current, speaker_b_name) if speaker_b_name else ""
                    chunk_list.append((current, "", b_chunk))
                current = part
        if current:
            b_chunk = _extract_speaker_lines(current, speaker_b_name) if speaker_b_name else ""
            chunk_list.append((current, "", b_chunk))
        chunks = chunk_list

    conv_ts = _parse_locomo_datetime(date_time)
    timestamp = conv_ts if conv_ts else int(time.time() * 1000)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    saved: list[dict] = []
    for i, chunk_tuple in enumerate(chunks):
        chunk_text, a_text, b_text = chunk_tuple
        user_text = chunk_text
        asst_text = b_text if b_text else chunk_text
        qa_content = json.dumps({"user": user_text, "assistant": asst_text})
        payload: dict[str, Any] = {
            "tenant_id": "default",
            "scope": "user",
            "session_id": f"agent:main:openresponses-user:{user_key}",
            "user_id": user_key,
            "content": qa_content,
            "type_hint": "qa_chunk",
            "source": "eval_ingest",
            "timestamp": timestamp,
        }

        # Exponential backoff retry
        for attempt in range(max_retries):
            try:
                async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
                    if not resp.ok:
                        error_text = await resp.text()
                        raise RuntimeError(f"API error {resp.status}: {error_text[:200]}")
                    result = await resp.json()
                    if not result.get("ok"):
                        raise RuntimeError(result.get("error", "save failed"))
                    saved.append({"chunk_index": i, "data": result.get("data")})
                    break
            except Exception as exc:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt), 10.0)
                    logger.warning(
                        "Ingest chunk %d failed (attempt %d/%d): %s, retrying in %.1fs",
                        i, attempt + 1, max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Ingest chunk %d failed after %d attempts: %s", i, max_retries, exc)
                    raise

    return saved


# ---------------------------------------------------------------------------
# Session cleanup — ported from original main() epilogue
# ---------------------------------------------------------------------------

def _cleanup_sessions() -> int:
    """Remove all files under ``~/.openclaw/agents/main/sessions/``.

    Returns the number of files removed.
    """
    sessions_dir = os.path.expanduser("~/.openclaw/agents/main/sessions")
    if not os.path.isdir(sessions_dir):
        return 0
    cleared = 0
    for entry in os.listdir(sessions_dir):
        fp = os.path.join(sessions_dir, entry)
        if os.path.isfile(fp):
            os.remove(fp)
            cleared += 1
    return cleared


# ---------------------------------------------------------------------------
# Resource
# ---------------------------------------------------------------------------

@register_resource("memsense_ingest")
class MemsenseIngestResource(BaseResource):
    """Ingest one sample's conversations into Memsense memory with async concurrency."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8787",
        token: str | None = None,
        max_chunk_size: int = 4000,
        user_prefix: str = "eval",
        concurrency: int = 3,
        max_retries: int = 3,
        base_delay: float = 2.0,
        cleanup_sessions: bool = True,
        **_kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.max_chunk_size = max_chunk_size
        self.user_prefix = user_prefix
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.cleanup_sessions = cleanup_sessions

    async def process(self, sample: dict) -> tuple:
        """Ingest all sessions for a single sample with concurrent session processing.

        *sample* is a dict with keys ``sample_id``, ``conversations``.
        Returns ``(ingest_result_dict,)``.
        """
        sample_id = sample["sample_id"]
        conversations: list[dict] = sample["conversations"]
        user_key = f"{self.user_prefix}-{sample_id}"

        sem = asyncio.Semaphore(self.concurrency)

        async def _ingest_session(sess: dict) -> dict:
            async with sem:
                meta = sess["meta"]
                msg = sess["message"]
                speaker_a_text = sess.get("speaker_a_text", "")
                speaker_b_text = sess.get("speaker_b_text", "")
                label = f"{meta['session_key']} ({meta['date_time']})"
                try:
                    async with aiohttp.ClientSession(trust_env=False) as http:
                        saved = await _save_to_memsense(
                            http,
                            self.base_url,
                            msg,
                            user_key,
                            meta["session_key"],
                            self.token,
                            self.max_chunk_size,
                            speaker_a_text,
                            speaker_b_text,
                            meta.get("date_time", ""),
                            meta.get("speaker_b", ""),
                            self.max_retries,
                            self.base_delay,
                        )
                    logger.info(
                        "[%s] %s — saved %d chunk(s)", sample_id, label, len(saved)
                    )
                    return {
                        "sample_id": sample_id,
                        "session": meta["session_key"],
                        "user": user_key,
                        "status": "success",
                        "chunks_count": len(saved),
                        "chunks": saved,
                    }
                except Exception as exc:
                    logger.error("[%s] %s — ERROR: %s", sample_id, label, exc)
                    return {
                        "sample_id": sample_id,
                        "session": meta["session_key"],
                        "user": user_key,
                        "status": "error",
                        "error": str(exc),
                    }

        tasks = [_ingest_session(sess) for sess in conversations]
        results = await asyncio.gather(*tasks)

        # Session cleanup — mirrors original ingest.py main() epilogue
        if self.cleanup_sessions:
            cleared = await asyncio.to_thread(_cleanup_sessions)
            if cleared:
                logger.info("[cleanup] cleared %d session file(s)", cleared)

        return ({"sample_id": sample_id, "user": user_key, "sessions": list(results)},)

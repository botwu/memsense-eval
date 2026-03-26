"""Ingest conversations into the Memsense memory API.

Extracted from the original ``ingest.py`` (save_to_memsense,
generate_tags_with_openclaw).
"""

from __future__ import annotations

import json
import logging
import subprocess
import re
import time
from typing import Any

import requests

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)


def _generate_tags_with_openclaw(content: str) -> dict:
    prompt = (
        "You are a background memory tagger. Return JSON only.\n"
        "Task: generate up to 8 concise tags, one memory_kind, and a brief summary.\n"
        'memory_kind must be exactly one of: stable, preference, episodic, ephemeral.\n'
        "Tags rules: lowercase, short noun/verb phrases, no punctuation noise, no duplicate synonyms.\n"
        "Summary: one concise sentence (max 100 chars).\n"
        'Output format: {"memory_kind": "preference", "tags": ["tag1", "tag2"], "summary": "brief summary"}\n'
        f"\nInput:\n{content}"
    )

    try:
        result = subprocess.run(
            [
                "openclaw", "agent",
                "--session-id", "memsense-tagger",
                "--message", prompt,
                "--json", "--timeout", "90",
            ],
            capture_output=True, text=True, timeout=95,
        )
        if result.returncode != 0:
            logger.warning("openclaw tag generation failed: %s", result.stderr[:100])
            return {"tags": [], "memory_kind": "episodic", "summary": None}

        data = json.loads(result.stdout)
        text = data.get("result", {}).get("payloads", [{}])[0].get("text", "")

        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            output = json.loads(match.group(1))
        else:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                output = json.loads(match.group(0))
            else:
                return {"tags": [], "memory_kind": "episodic", "summary": None}

        return {
            "tags": output.get("tags", [])[:8],
            "memory_kind": output.get("memory_kind", "episodic"),
            "summary": output.get("summary"),
        }
    except Exception as exc:
        logger.warning("Tag generation error: %s", str(exc)[:100])
        return {"tags": [], "memory_kind": "episodic", "summary": None}


def _save_to_memsense(
    base_url: str,
    content: str,
    user_key: str,
    session_key: str,
    token: str | None = None,
    generate_tags: bool = False,
    max_chunk_size: int = 4000,
) -> list[dict]:
    url = f"{base_url}/v1/memory/save"

    # Chunk long content
    if len(content) <= max_chunk_size:
        chunks = [content]
    else:
        parts = content.split("\n\n")
        chunks: list[str] = []
        current = ""
        for part in parts:
            if len(current) + len(part) + 2 <= max_chunk_size:
                current += ("\n\n" if current else "") + part
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    saved: list[dict] = []
    for i, chunk in enumerate(chunks):
        qa_content = json.dumps({"user": chunk, "assistant": ""})
        payload: dict[str, Any] = {
            "tenant_id": "default",
            "scope": "user",
            "session_id": f"agent:main:openresponses-user:{user_key}",
            "user_id": user_key,
            "content": qa_content,
            "type_hint": "qa_chunk",
            "source": "eval_ingest",
            "timestamp": int(time.time() * 1000),
        }

        if generate_tags:
            tag_data = _generate_tags_with_openclaw(chunk)
            payload["tags"] = tag_data["tags"]
            payload["task_tag"] = tag_data["summary"]
            logger.debug("Tags: %s", tag_data["tags"])

        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")
        result = resp.json()
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "save failed"))
        saved.append({"chunk_index": i, "data": result.get("data")})

    return saved


@register_resource("memsense_ingest")
class MemsenseIngestResource(BaseResource):
    """Ingest one sample's conversations into Memsense memory."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8787",
        token: str | None = None,
        generate_tags: bool = False,
        max_chunk_size: int = 4000,
        user_prefix: str = "eval",
        **_kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.token = token
        self.generate_tags = generate_tags
        self.max_chunk_size = max_chunk_size
        self.user_prefix = user_prefix

    async def process(self, sample: dict) -> tuple:
        """Ingest all sessions for a single sample.

        *sample* is a dict with keys ``sample_id``, ``conversations``.
        Returns ``(ingest_result_dict,)``.
        """
        sample_id = sample["sample_id"]
        conversations: list[dict] = sample["conversations"]
        user_key = f"{self.user_prefix}-{sample_id}"

        results: list[dict] = []
        for sess in conversations:
            meta = sess["meta"]
            msg = sess["message"]
            label = f"{meta['session_key']} ({meta['date_time']})"
            try:
                saved = _save_to_memsense(
                    self.base_url,
                    msg,
                    user_key,
                    meta["session_key"],
                    self.token,
                    self.generate_tags,
                    self.max_chunk_size,
                )
                logger.info(
                    "[%s] %s — saved %d chunk(s)", sample_id, label, len(saved)
                )
                results.append({
                    "sample_id": sample_id,
                    "session": meta["session_key"],
                    "user": user_key,
                    "status": "success",
                    "chunks_count": len(saved),
                    "chunks": saved,
                })
            except Exception as exc:
                logger.error("[%s] %s — ERROR: %s", sample_id, label, exc)
                results.append({
                    "sample_id": sample_id,
                    "session": meta["session_key"],
                    "user": user_key,
                    "status": "error",
                    "error": str(exc),
                })

        return ({"sample_id": sample_id, "user": user_key, "sessions": results},)

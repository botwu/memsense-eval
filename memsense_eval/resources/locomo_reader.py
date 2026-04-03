"""Read a LoCoMo JSON file and produce structured per-sample cases.

Extracted from the original ``ingest.py`` (load_locomo_data,
build_session_messages) and ``qa.py`` (QA filtering).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)


def _format_message(msg: dict) -> str:
    speaker = msg.get("speaker", "unknown")
    text = msg.get("text", "")
    line = f"{speaker}: {text}"

    img_urls = msg.get("img_url", [])
    if isinstance(img_urls, str):
        img_urls = [img_urls]
    blip = msg.get("blip_caption", "")

    if img_urls:
        for url in img_urls:
            caption = f": {blip}" if blip else ""
            line += f"\n{url}{caption}"
    elif blip:
        line += f"\n({blip})"

    return line


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


def _build_session_messages(
    item: dict,
    session_range: tuple[int, int] | None = None,
    head: str = "",
    tail: str = "",
) -> list[dict]:
    conv = item["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    speakers = f"{speaker_a} & {speaker_b}"

    session_keys = sorted(
        [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
        key=lambda k: int(k.split("_")[1]),
    )

    sessions: list[dict] = []
    for sk in session_keys:
        sess_num = int(sk.split("_")[1])
        if session_range:
            lo, hi = session_range
            if sess_num < lo or sess_num > hi:
                continue

        dt_key = f"{sk}_date_time"
        date_time = conv.get(dt_key, "")

        header = f"[group chat conversation: {date_time}]"
        parts: list[str] = []
        parts_a: list[str] = []
        parts_b: list[str] = []
        if head:
            parts.append(head)
            parts_a.append(head)
            parts_b.append(head)
        parts.append(header)
        parts_a.append(header)
        parts_b.append(header)
        for msg in conv[sk]:
            formatted = _format_message(msg)
            parts.append(formatted)
            speaker = msg.get("speaker", "")
            if speaker == speaker_a:
                parts_a.append(formatted)
            elif speaker == speaker_b:
                parts_b.append(formatted)
            else:
                parts_a.append(formatted)
                parts_b.append(formatted)
        if tail:
            parts.append(tail)
            parts_a.append(tail)
            parts_b.append(tail)

        sessions.append({
            "message": "\n\n".join(parts),
            "speaker_a_text": "\n\n".join(parts_a),
            "speaker_b_text": "\n\n".join(parts_b),
            "meta": {
                "sample_id": item["sample_id"],
                "session_key": sk,
                "date_time": date_time,
                "speakers": speakers,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            },
        })

    return sessions


@register_resource("locomo_reader")
class LocomoReaderResource(BaseResource):
    """Read LoCoMo JSON → ``{sample_id: {conversations, qa_list, ...}}``."""

    def __init__(
        self,
        filter_category: str = "5",
        session_range: str | None = None,
        head: str = "",
        tail: str = "",
        **_kwargs: Any,
    ) -> None:
        self.filter_category = filter_category
        self.session_range = self._parse_range(session_range) if session_range else None
        self.head = head
        self.tail = tail

    @staticmethod
    def _parse_range(s: str) -> tuple[int, int]:
        if "-" in s:
            lo, hi = s.split("-", 1)
            return int(lo), int(hi)
        n = int(s)
        return n, n

    async def process(self, data_path: str) -> tuple:  # noqa: D401
        """Returns ``(samples_dict,)`` keyed by stringified index."""
        with open(data_path, "r", encoding="utf-8") as f:
            raw: list[dict] = json.load(f)

        samples: dict[str, dict] = {}
        for idx, item in enumerate(raw):
            conversations = _build_session_messages(
                item,
                session_range=self.session_range,
                head=self.head,
                tail=self.tail,
            )
            qa_list = [
                q
                for q in item.get("qa", [])
                if str(q.get("category", "")) != self.filter_category
            ]

            samples[str(idx)] = {
                "sample_id": item["sample_id"],
                "conversations": conversations,
                "qa_list": qa_list,
            }

        logger.info(
            "Loaded %d samples (%d total QA) from %s",
            len(samples),
            sum(len(s["qa_list"]) for s in samples.values()),
            data_path,
        )
        return (samples,)

"""Wait for all embedding and tag jobs to complete after ingest.

The memsense pipeline is:
    saveChunk() → embedding_job(full) + tag_job
    tag-worker  → processes tag_job → creates embedding_job(user) + embedding_job(assistant)
                                    → writes facets → creates embedding_job(facet, facet_type=*)
    embedding-worker → processes all embedding_jobs

We must wait for **both** tag_jobs AND embedding_jobs to finish, because
tag-worker creates secondary embedding_jobs.  If we only waited for
embedding_jobs, we could see "0 pending" before tag-worker has created
the user/assistant/facet jobs.

**Notification mode (default):** Workers fire ``NOTIFY memsense_jobs``
after each job completion.  This resource uses ``LISTEN memsense_jobs``
so it wakes up immediately instead of polling.  A fallback poll every
*poll_interval* seconds guards against missed notifications.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import asyncpg

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)

_POLL_SQL = """
SELECT 'embedding' AS job_type, ej.status, COUNT(*) AS cnt
FROM embedding_jobs ej
JOIN memory_chunks mc ON ej.chunk_id = mc.id
WHERE mc.user_id = $1
GROUP BY ej.status
UNION ALL
SELECT 'tag' AS job_type, tj.status, COUNT(*) AS cnt
FROM tag_jobs tj
JOIN memory_chunks mc ON tj.chunk_id = mc.id
WHERE mc.user_id = $1
GROUP BY tj.status
"""

_CHANNEL = "memsense_jobs"


def _parse_status(rows):
    """Parse poll query rows into (emb_status, tag_status) dicts."""
    emb: dict[str, int] = {}
    tag: dict[str, int] = {}
    for r in rows:
        bucket = emb if r["job_type"] == "embedding" else tag
        bucket[r["status"]] = r["cnt"]
    return emb, tag


@register_resource("embedding_wait")
class EmbeddingWaitResource(BaseResource):
    """Block until all tag + embedding jobs for a user_id are done.

    Uses PostgreSQL LISTEN/NOTIFY for instant wakeup when workers
    complete jobs.  Falls back to periodic polling as a safety net.
    """

    def __init__(
        self,
        db_url: str = "postgresql://127.0.0.1:5432/memsense",
        poll_interval: float = 30.0,
        timeout: float = 300.0,
        **_kwargs: Any,
    ) -> None:
        self.db_url = db_url
        self.poll_interval = poll_interval
        self.timeout = timeout

    async def process(self, ingest_result: dict) -> tuple:
        """Wait for tag-worker + embedding-worker to finish for the given ingest.

        *ingest_result* must contain ``user`` (the user_id used in ingest)
        and ``sample_id``.
        Returns ``(ready_dict,)`` once all jobs are done.
        """
        user_id: str = ingest_result["user"]
        sample_id: str = ingest_result["sample_id"]

        conn: asyncpg.Connection = await asyncpg.connect(self.db_url)
        try:
            # Event set by LISTEN callback — any worker completion wakes us
            notify_event = asyncio.Event()

            def _on_notify(
                _conn: asyncpg.Connection,
                _pid: int,
                _channel: str,
                _payload: str,
            ) -> None:
                notify_event.set()

            await conn.add_listener(_CHANNEL, _on_notify)

            t0 = time.monotonic()
            emb_status: dict[str, int] = {}
            tag_status: dict[str, int] = {}

            while True:
                elapsed = time.monotonic() - t0
                if elapsed >= self.timeout:
                    break

                rows = await conn.fetch(_POLL_SQL, user_id)
                emb_status, tag_status = _parse_status(rows)

                emb_total = sum(emb_status.values())
                tag_total = sum(tag_status.values())

                if emb_total == 0 and tag_total == 0:
                    logger.warning(
                        "[%s] No jobs found for user_id=%s, continuing anyway",
                        sample_id, user_id,
                    )
                    return ({"sample_id": sample_id, "status": "no_jobs", "total": 0},)

                emb_done = emb_status.get("done", 0)
                emb_pending = emb_status.get("pending", 0)
                emb_running = emb_status.get("running", 0)
                emb_failed = emb_status.get("failed", 0)

                tag_done = tag_status.get("done", 0)
                tag_pending = tag_status.get("pending", 0)
                tag_running = tag_status.get("running", 0)

                tags_done = tag_pending == 0 and tag_running == 0
                embeds_done = emb_pending == 0 and emb_running == 0

                if tags_done and embeds_done:
                    logger.info(
                        "[%s] All jobs finished — tags: %d done, embeddings: %d done (%d failed)",
                        sample_id, tag_done, emb_done, emb_failed,
                    )
                    return ({
                        "sample_id": sample_id,
                        "status": "ready",
                        "embedding_total": emb_total,
                        "embedding_done": emb_done,
                        "embedding_failed": emb_failed,
                        "tag_total": tag_total,
                        "tag_done": tag_done,
                    },)

                logger.debug(
                    "[%s] Waiting: tags(total=%d done=%d pending=%d running=%d) "
                    "emb(total=%d done=%d pending=%d running=%d) [%.0fs]",
                    sample_id,
                    tag_total, tag_done, tag_pending, tag_running,
                    emb_total, emb_done, emb_pending, emb_running,
                    elapsed,
                )

                # Wait for NOTIFY or fallback poll_interval, whichever comes first
                notify_event.clear()
                remaining = self.timeout - elapsed
                wait_sec = min(self.poll_interval, remaining)
                try:
                    await asyncio.wait_for(notify_event.wait(), timeout=wait_sec)
                except asyncio.TimeoutError:
                    pass  # fallback poll

            # Timeout
            logger.error(
                "[%s] Wait timed out after %.0fs — tags(pending=%d running=%d) emb(pending=%d running=%d)",
                sample_id, self.timeout,
                tag_status.get("pending", 0), tag_status.get("running", 0),
                emb_status.get("pending", 0), emb_status.get("running", 0),
            )
            return ({
                "sample_id": sample_id,
                "status": "timeout",
                "embedding_total": sum(emb_status.values()),
                "embedding_done": emb_status.get("done", 0),
                "embedding_pending": emb_status.get("pending", 0),
                "tag_total": sum(tag_status.values()),
                "tag_pending": tag_status.get("pending", 0),
            },)
        finally:
            try:
                await conn.remove_listener(_CHANNEL, _on_notify)
            except Exception:
                pass
            await conn.close()

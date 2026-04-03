"""LLM-as-judge grading of QA results.

Enhanced with exponential backoff retry strategy and diskcache for LLM response caching.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
import diskcache

from memsense_eval.engine.resource import BaseResource, register_resource

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions "
    "match a gold standard answer"
)

_ACCURACY_TEMPLATE = """\
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. \
You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know \
about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes \
the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous \
with your grading - as long as it touches on the same topic as the gold \
answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, \
month, year, etc. The generated answer might be much longer or use \
relative time references (like "last Tuesday" or "next month"), but you \
should be generous with your grading - as long as it refers to the same \
date or time period as the gold answer, it should be counted as CORRECT. \
Even if the format differs (e.g., "May 7th" vs "7 May"), consider it \
CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, \
then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will \
break the evaluation script.

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
"""


def _cache_key(question: str, gold_answer: str, response: str) -> str:
    """Generate a stable cache key for the grading request."""
    content = f"{question}|||{gold_answer}|||{response}"
    return hashlib.sha256(content.encode()).hexdigest()


async def _locomo_grader(
    client: AsyncOpenAI,
    model: str,
    question: str,
    gold_answer: str,
    response: str,
    cache: diskcache.Cache | None = None,
    max_retries: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
) -> bool:
    # Check cache first
    if cache is not None:
        cache_key = _cache_key(question, gold_answer, response)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for question: %s", question[:60])
            return cached

    prompt = _ACCURACY_TEMPLATE.format(
        question=question, gold_answer=gold_answer, response=response
    )

    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = resp.choices[0].message.content or ""

            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # LLM returned non-JSON — fall back to keyword detection
                upper = content.upper()
                if "CORRECT" in upper and "WRONG" not in upper:
                    is_correct = True
                elif "WRONG" in upper and "CORRECT" not in upper:
                    is_correct = False
                else:
                    logger.warning(
                        "Judge returned unparseable response (attempt %d): %s",
                        attempt + 1, content[:120],
                    )
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        await asyncio.sleep(delay)
                        continue
                    is_correct = False
            else:
                label = result.get("is_correct", result.get("label", "WRONG"))
                is_correct = label.strip().lower() == "correct"

            # Cache the result
            if cache is not None:
                cache_key = _cache_key(question, gold_answer, response)
                cache.set(cache_key, is_correct)

            return is_correct

        except RateLimitError:
            if attempt >= max_retries - 1:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning("Rate limit hit (attempt %d/%d), retrying in %.1fs", attempt + 1, max_retries, delay)
            await asyncio.sleep(delay)
        except Exception as exc:
            logger.warning(
                "Judge API error (attempt %d/%d): %s",
                attempt + 1, max_retries, exc,
            )
            if attempt >= max_retries - 1:
                logger.error("Judge gave up after %d attempts: %s", max_retries, question[:60])
                return False
            delay = min(base_delay * (2 ** attempt), max_delay)
            await asyncio.sleep(delay)

    return False


@register_resource("llm_judge")
class LLMJudgeResource(BaseResource):
    """Grade a list of QA records using an LLM judge with caching."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 5,
        cache_dir: str | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        **_kwargs: Any,
    ) -> None:
        load_dotenv()
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Initialize diskcache
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(cache_dir)
            logger.info("Judge cache enabled at %s", cache_dir)
        else:
            self.cache = None

    async def process(self, qa_results: list[dict]) -> tuple:
        """Grade every QA record with caching.

        Returns ``(graded_list,)`` where each entry gains a ``grade`` bool.
        """
        sem = asyncio.Semaphore(self.concurrency)

        async def _grade_one(item: dict) -> dict:
            async with sem:
                is_correct = await _locomo_grader(
                    self.client,
                    self.model,
                    item["question"],
                    item["expected"],
                    item["response"],
                    self.cache,
                    self.max_retries,
                    self.base_delay,
                    self.max_delay,
                )
                return {**item, "grade": is_correct}

        tasks = [_grade_one(item) for item in qa_results]
        graded = await asyncio.gather(*tasks)

        correct = sum(1 for g in graded if g["grade"])
        logger.info(
            "Judge: %d/%d correct (%.1f%%)",
            correct, len(graded), 100 * correct / len(graded) if graded else 0,
        )
        return (list(graded),)


@register_resource("llm_judge_single")
class LLMJudgeSingleResource(LLMJudgeResource):
    """Grade a single QA record, dispatched per-question for streaming."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._sem = asyncio.Semaphore(self.concurrency)

    async def process(self, qa_item: dict) -> tuple:  # type: ignore[override]
        """Grade one QA record. Returns ``(graded_item,)``."""
        async with self._sem:
            is_correct = await _locomo_grader(
                self.client,
                self.model,
                qa_item["question"],
                qa_item["expected"],
                qa_item["response"],
                self.cache,
                self.max_retries,
                self.base_delay,
                self.max_delay,
            )
            return ({**qa_item, "grade": is_correct},)

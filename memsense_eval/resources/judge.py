"""LLM-as-judge grading of QA results.

Extracted from the original ``judge_util.py`` (locomo_grader) and
``judge.py`` (grade_answers_with_concurrency).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

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


async def _locomo_grader(
    client: AsyncOpenAI,
    model: str,
    question: str,
    gold_answer: str,
    response: str,
    max_retries: int = 5,
) -> bool:
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
                    return True
                if "WRONG" in upper and "CORRECT" not in upper:
                    return False
                logger.warning(
                    "Judge returned unparseable response (attempt %d): %s",
                    attempt + 1, content[:120],
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                return False

            label = result.get("is_correct", result.get("label", "WRONG"))
            return label.strip().lower() == "correct"

        except RateLimitError:
            if attempt >= max_retries - 1:
                raise
            await asyncio.sleep(0.5 * (attempt + 1))
        except Exception as exc:
            logger.warning(
                "Judge API error (attempt %d/%d): %s",
                attempt + 1, max_retries, exc,
            )
            if attempt >= max_retries - 1:
                logger.error("Judge gave up after %d attempts: %s", max_retries, question[:60])
                return False
            await asyncio.sleep(0.5 * (attempt + 1))

    return False


@register_resource("llm_judge")
class LLMJudgeResource(BaseResource):
    """Grade a list of QA records using an LLM judge."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 5,
        **_kwargs: Any,
    ) -> None:
        load_dotenv()
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )
        self.concurrency = concurrency

    async def process(self, qa_results: list[dict]) -> tuple:
        """Grade every QA record.

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

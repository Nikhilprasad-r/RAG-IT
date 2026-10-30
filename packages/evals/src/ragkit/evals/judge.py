"""An LLM judge for answers that cannot be checked by exact comparison."""

from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from ragkit.core.types import ScoredChunk

CORRECTNESS = """You grade an AI assistant's answer against a reference answer.

Question:
{question}

Reference answer:
{reference}

Assistant's answer:
{answer}

Score 1.0 if the answer states the same facts as the reference (wording, order and extra \
helpful detail do not matter), 0.5 if it is partly right or misses something important, and \
0.0 if it is wrong, contradicts the reference, or does not answer. If the reference says the \
assistant should decline or say it does not know, score 1.0 when it does so."""

GROUNDEDNESS = """You check whether an AI assistant's answer is supported by the passages \
it was given.

Passages:
{passages}

Question:
{question}

Assistant's answer:
{answer}

Score the fraction of factual claims in the answer that the passages support, from 0.0 to \
1.0. Statements that the information is not available, and greetings, count as supported."""


class Verdict(BaseModel):
    score: float = Field(ge=0, le=1, description="0.0 to 1.0")
    reason: str = Field(description="One sentence explaining the score.")


class Judge:
    def __init__(self, model: str | BaseChatModel) -> None:
        chat = init_chat_model(model) if isinstance(model, str) else model
        self.model_name = model if isinstance(model, str) else type(model).__name__
        self._grader = chat.with_structured_output(Verdict)

    async def correctness(self, question: str, answer: str, reference: str) -> Verdict:
        prompt = CORRECTNESS.format(question=question, reference=reference, answer=answer)
        return await self._grade(prompt)

    async def groundedness(
        self, question: str, answer: str, passages: Sequence[ScoredChunk]
    ) -> Verdict:
        context = "\n\n".join(
            f"[{n}] {p.title or p.uri}\n{p.chunk.text}" for n, p in enumerate(passages, start=1)
        )
        prompt = GROUNDEDNESS.format(passages=context, question=question, answer=answer)
        return await self._grade(prompt)

    async def _grade(self, prompt: str) -> Verdict:
        verdict = await self._grader.ainvoke(prompt)
        return verdict if isinstance(verdict, Verdict) else Verdict.model_validate(verdict)

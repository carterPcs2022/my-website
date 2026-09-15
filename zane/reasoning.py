"""Provider-independent reasoning boundary for Zane.

Zane's identity, memory, knowledge, skills, and safety systems must not depend
on a particular cloud model. This interface lets a local model become the
primary reasoning engine later while cloud providers remain optional adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .knowledge import KnowledgeVault


@dataclass(frozen=True)
class ReasoningResult:
    text: str
    mode: str
    knowledge_used: int = 0


class ReasoningBackend(Protocol):
    def generate(self, prompt: str, context: str = "") -> str: ...


class OfflineReasoning:
    """Deterministic fallback that requires no network or model provider."""

    def __init__(self, knowledge: KnowledgeVault) -> None:
        self.knowledge = knowledge

    def answer(self, prompt: str) -> ReasoningResult:
        matches = self.knowledge.search(prompt, limit=3)
        if matches:
            context = "\n".join(f"{item.title}: {item.content}" for item in matches)
            return ReasoningResult(
                text=f"Local knowledge retrieved:\n{context}",
                mode="offline-knowledge",
                knowledge_used=len(matches),
            )
        return ReasoningResult(
            text="No matching local knowledge was found. A local or cloud reasoning backend can be attached.",
            mode="offline-no-model",
        )


class ZaneReasoning:
    """Stable reasoning facade; the backend is replaceable."""

    def __init__(self, knowledge: KnowledgeVault, backend: ReasoningBackend | None = None) -> None:
        self.knowledge = knowledge
        self.offline = OfflineReasoning(knowledge)
        self.backend = backend

    def answer(self, prompt: str, allow_cloud: bool = False) -> ReasoningResult:
        if not allow_cloud or self.backend is None:
            return self.offline.answer(prompt)

        matches = self.knowledge.search(prompt, limit=5)
        context = "\n".join(f"{item.title}: {item.content}" for item in matches)
        text = self.backend.generate(prompt, context=context)
        return ReasoningResult(text=text, mode="external-backend", knowledge_used=len(matches))

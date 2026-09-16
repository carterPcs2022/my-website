"""Offline-first knowledge vault for Zane.

Knowledge is deliberately separate from episodic memory and from the LLM
provider. A future SQLite/vector backend can replace this implementation
without changing Zane's identity layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class KnowledgeItem:
    title: str
    content: str
    category: str = "general"
    source: str = "local"


class KnowledgeVault:
    def __init__(self, root: str | Path = "knowledge") -> None:
        self.root = Path(root)
        self.items: list[KnowledgeItem] = []

    def add(self, item: KnowledgeItem) -> None:
        self.items.append(item)

    def load_json(self, path: str | Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        records = data.get("items", data) if isinstance(data, dict) else data
        added = 0
        for record in records:
            self.add(KnowledgeItem(**record))
            added += 1
        return added

    def load_text(self, path: str | Path, category: str = "lore") -> int:
        """Load a plain-text knowledge source as paragraph-sized items.

        This keeps the offline reasoning layer useful when a deployment has
        raw lore/spec text but has not yet built the FAISS archival index.
        """
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        added = 0
        for index, paragraph in enumerate(
            (part.strip() for part in text.split("\n\n")), start=1
        ):
            if not paragraph:
                continue
            self.add(
                KnowledgeItem(
                    title=f"{category.title()} {index}",
                    content=paragraph,
                    category=category,
                    source=str(source),
                )
            )
            added += 1
        return added

    def search(self, query: str, limit: int = 5) -> list[KnowledgeItem]:
        terms = set(re.findall(r"[a-z0-9]+", str(query).lower()))
        if not terms:
            return []
        scored: list[tuple[int, KnowledgeItem]] = []
        for item in self.items:
            haystack = f"{item.title} {item.content} {item.category}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[: max(0, int(limit))]]

    def snapshot(self) -> list[dict]:
        return [asdict(item) for item in self.items]

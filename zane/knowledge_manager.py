"""Multi-namespace local RAG: a static, immutable "archival lore" index
searched alongside Zane's organic chat memory, without the two ever
mixing.

`KnowledgeManager` also exposes the provider-independent `ZaneReasoning`
facade. This gives the actual ZaneMind request path a deterministic local
reasoning/knowledge fallback even when the archival FAISS index has not
been compiled yet.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from zane.knowledge import KnowledgeVault
from zane.reasoning import ZaneReasoning
from zane.memory.embeddings import EmbeddingBackend, EmbeddingError
from zane.memory.vector_index import FaissVectorIndex, VectorIndexError

if TYPE_CHECKING:
    import numpy as np

    from zane.memory.persistent import PersistentMemory

logger = logging.getLogger("zane.knowledge_manager")

DEFAULT_LORE_INDEX_PATH = "data/ninjago_lore.index"
DEFAULT_LORE_METADATA_PATH = "data/ninjago_lore.json"
DEFAULT_RAW_LORE_PATH = "data/raw_lore.txt"

LORE_TAG = "[ARCHIVAL_LORE_DATABASE]"
HISTORY_TAG = "[HISTORICAL_CONTEXT]"
REASONING_TAG = "[OFFLINE_REASONING_CONTEXT]"


@dataclass
class LoreChunk:
    id: int
    text: str
    source: str
    chunk_index: int


class KnowledgeManager:
    def __init__(
        self,
        embeddings: EmbeddingBackend,
        lore_index_path: str = DEFAULT_LORE_INDEX_PATH,
        lore_metadata_path: str = DEFAULT_LORE_METADATA_PATH,
    ) -> None:
        # Reuses the SAME embedding model instance chat memory already
        # loaded — one model in memory, not two.
        self.embeddings = embeddings
        self.lore_metadata_path = lore_metadata_path
        self._lore_vector_index = FaissVectorIndex(lore_index_path)
        self._lore_chunks: Dict[int, LoreChunk] = self._load_lore_metadata()

        # Provider-independent reasoning boundary. It is intentionally
        # separate from the FAISS namespace so it can still operate when
        # the archival index has not been compiled on a fresh deployment.
        self.offline_vault = KnowledgeVault()
        raw_lore = Path(DEFAULT_RAW_LORE_PATH)
        if raw_lore.exists():
            try:
                loaded = self.offline_vault.load_text(raw_lore)
                logger.info("Loaded %d raw-lore items into offline reasoning vault.", loaded)
            except OSError as exc:
                logger.warning("Failed to load raw lore for offline reasoning: %s", exc)
        self.reasoning = ZaneReasoning(self.offline_vault)

    def _load_lore_metadata(self) -> Dict[int, LoreChunk]:
        path = Path(self.lore_metadata_path)
        if not path.exists():
            logger.info(
                "No lore metadata at %s yet; archival lore database is empty until "
                "compile_lore_database() has been run.", path,
            )
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load lore metadata from %s: %s", path, exc)
            return {}

        chunks: Dict[int, LoreChunk] = {}
        for key, value in raw.get("chunks", {}).items():
            try:
                chunks[int(key)] = LoreChunk(**value)
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed lore chunk %r: %s", key, exc)
        logger.info("Loaded %d archival lore chunks from %s.", len(chunks), path)
        return chunks

    async def _query_lore_tagged(self, query_vector: "np.ndarray", top_k: int) -> List[str]:
        try:
            hits = await asyncio.to_thread(self._lore_vector_index.search, query_vector, top_k)
        except VectorIndexError as exc:
            logger.warning("Archival lore search failed; skipping this turn: %s", exc)
            return []
        return [
            f"{LORE_TAG} {self._lore_chunks[vec_id].text}"
            for vec_id, _score in hits
            if vec_id in self._lore_chunks
        ]

    async def _query_history_tagged(
        self,
        user_memory: Optional["PersistentMemory"],
        query_text: str,
        query_vector: "np.ndarray",
        top_k: int,
    ) -> List[str]:
        if user_memory is None:
            return []
        hits = await user_memory.retrieve_relevant(query_text, top_k=top_k, query_vector=query_vector)
        return [f"{HISTORY_TAG} {h}" for h in hits]

    async def query_all_knowledge_sources(
        self,
        query_text: str,
        user_memory: Optional["PersistentMemory"] = None,
        top_k: int = 3,
    ) -> List[str]:
        """Search archival lore + history and inject the repaired offline
        reasoning context into the same list consumed by ZaneMind.respond().

        Embedding failure still degrades to an empty RAG result, while the
        provider-independent reasoning layer is deliberately independent of
        embeddings and can continue to supply local context.
        """
        if not query_text or not query_text.strip():
            return []

        reasoning_result = self.reasoning.answer(query_text, allow_cloud=False)
        reasoning_hits: List[str] = []
        if reasoning_result.knowledge_used:
            reasoning_hits.append(f"{REASONING_TAG} {reasoning_result.text}")

        try:
            query_vector = await asyncio.to_thread(self.embeddings.embed, query_text)
        except EmbeddingError as exc:
            logger.warning(
                "Knowledge query embedding failed; skipping lore/history lookups this turn: %s", exc
            )
            return reasoning_hits

        lore_hits, history_hits = await asyncio.gather(
            self._query_lore_tagged(query_vector, top_k),
            self._query_history_tagged(user_memory, query_text, query_vector, top_k),
        )
        return reasoning_hits + lore_hits + history_hits

    @property
    def lore_chunk_count(self) -> int:
        return len(self._lore_chunks)


# --------------------------------------------------------------------------
# Offline lore compilation — a CLI build step, not part of the runtime path.
# --------------------------------------------------------------------------


def _split_into_chunks(text: str, chunk_size_chars: int = 800, overlap_chars: int = 100) -> List[str]:
    """Deterministic chunking: greedily packs whole paragraphs (split on
    blank lines) up to `chunk_size_chars`, and falls back to fixed-size
    overlapping windows only for a single paragraph that alone exceeds
    the limit. Same input always produces the same chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size_chars:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(paragraph):
                end = start + chunk_size_chars
                chunks.append(paragraph[start:end])
                next_start = end - overlap_chars
                start = next_start if next_start > start else end
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)
    return chunks


def compile_lore_database(
    source_text_file: str,
    index_path: str = DEFAULT_LORE_INDEX_PATH,
    metadata_path: str = DEFAULT_LORE_METADATA_PATH,
    embeddings: Optional[EmbeddingBackend] = None,
    chunk_size_chars: int = 800,
    chunk_overlap_chars: int = 100,
) -> int:
    """Reads a raw text dump (show lore, engineering specs, etc.), splits
    it into deterministic chunks, embeds each chunk locally, and
    builds/overwrites the static `ninjago_lore` FAISS index + JSON
    metadata on disk. A one-time/occasional offline build step — never
    called from the request-handling path. Returns the number of chunks
    written. Raises FileNotFoundError/ValueError on bad input rather than
    silently producing an empty or partial database, since an operator
    running this manually wants to know immediately if it failed."""
    source_path = Path(source_text_file)
    if not source_path.exists():
        raise FileNotFoundError(f"Lore source file not found: {source_text_file}")

    text = source_path.read_text(encoding="utf-8")
    chunks = _split_into_chunks(text, chunk_size_chars, chunk_overlap_chars)
    if not chunks:
        raise ValueError(f"No content to chunk in {source_text_file} (file is empty or whitespace-only).")

    embeddings = embeddings or EmbeddingBackend()
    vector_index = FaissVectorIndex(index_path)

    metadata: Dict[str, dict] = {}
    for i, chunk_text in enumerate(chunks):
        chunk_id = i + 1
        vector = embeddings.embed(chunk_text)
        vector_index.add(chunk_id, vector)
        record = LoreChunk(id=chunk_id, text=chunk_text, source=str(source_path), chunk_index=i)
        metadata[str(chunk_id)] = asdict(record)

    metadata_path_obj = Path(metadata_path)
    metadata_path_obj.parent.mkdir(parents=True, exist_ok=True)
    metadata_path_obj.write_text(json.dumps({"chunks": metadata}, indent=2), encoding="utf-8")

    logger.info(
        "[ARCHIVAL_LORE_DATABASE] Compiled %d chunks from %s -> %s / %s",
        len(chunks), source_text_file, index_path, metadata_path,
    )
    return len(chunks)


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compile a raw text dump into Zane's static archival lore database."
    )
    parser.add_argument("source_text_file", help="Path to a raw text file (lore, specs, etc.).")
    parser.add_argument("--index-path", default=DEFAULT_LORE_INDEX_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_LORE_METADATA_PATH)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    count = compile_lore_database(
        args.source_text_file,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        chunk_size_chars=args.chunk_size,
        chunk_overlap_chars=args.chunk_overlap,
    )
    print(f"Compiled {count} lore chunks -> {args.index_path} / {args.metadata_path}")


if __name__ == "__main__":
    _main()

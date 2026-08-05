"""FAISS-backed vector index for Zane's retrieval-augmented memory.

Vector IDs are the same integer primary keys as `messages.id` in
`zane/memory/store.py`, so a similarity search result maps directly back
to a row without a separate ID-translation table. The index persists to
disk after every mutation for durability across restarts, and every
operation degrades gracefully on an empty/cold-start index rather than
raising.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("zane.memory.vector_index")


class VectorIndexError(RuntimeError):
    """Raised when a vector index operation fails."""


class FaissVectorIndex:
    def __init__(self, index_path: str) -> None:
        self._path = index_path
        self._lock = threading.RLock()
        self._index = None  # created lazily once the embedding dimension is known

        if os.path.exists(index_path):
            try:
                import faiss

                self._index = faiss.read_index(index_path)
            except ImportError as exc:
                raise VectorIndexError(
                    "faiss-cpu is not installed. Run `pip install faiss-cpu` to enable "
                    "persistent, retrieval-augmented memory."
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load existing FAISS index at %s (%s); starting fresh.",
                    index_path, exc,
                )
                self._index = None

    def _ensure_index(self, dim: int) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise VectorIndexError(
                "faiss-cpu is not installed. Run `pip install faiss-cpu` to enable "
                "persistent, retrieval-augmented memory."
            ) from exc
        # Inner product over L2-normalized vectors == cosine similarity.
        base = faiss.IndexFlatIP(dim)
        self._index = faiss.IndexIDMap2(base)

    def add(self, vector_id: int, vector: "np.ndarray") -> None:
        with self._lock:
            try:
                import numpy as np

                self._ensure_index(len(vector))
                v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
                ids = np.asarray([vector_id], dtype=np.int64)
                self._index.add_with_ids(v, ids)
                self._persist_locked()
            except VectorIndexError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise VectorIndexError(f"Failed to add vector {vector_id}: {exc}") from exc

    def search(self, vector: "np.ndarray", top_k: int) -> List[Tuple[int, float]]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0 or top_k <= 0:
                return []
            try:
                import numpy as np

                v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
                k = min(top_k, self._index.ntotal)
                scores, ids = self._index.search(v, k)
                return [
                    (int(vec_id), float(score))
                    for vec_id, score in zip(ids[0], scores[0])
                    if vec_id != -1
                ]
            except Exception as exc:  # noqa: BLE001
                raise VectorIndexError(f"Vector search failed: {exc}") from exc

    def remove(self, vector_ids: Sequence[int]) -> None:
        if not vector_ids:
            return
        with self._lock:
            if self._index is None:
                return
            try:
                import numpy as np

                self._index.remove_ids(np.asarray(list(vector_ids), dtype=np.int64))
                self._persist_locked()
            except Exception as exc:  # noqa: BLE001
                raise VectorIndexError(f"Failed to remove vectors {vector_ids}: {exc}") from exc

    def _persist_locked(self) -> None:
        try:
            import faiss

            faiss.write_index(self._index, self._path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist FAISS index to %s: %s", self._path, exc)

    @property
    def size(self) -> int:
        with self._lock:
            return 0 if self._index is None else self._index.ntotal

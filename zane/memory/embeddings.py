"""Local text embeddings for Zane's retrieval-augmented memory.

Uses `sentence-transformers` (all-MiniLM-L6-v2 by default) so embedding
computation never depends on an external API — it runs entirely on the
same machine as the rest of Zane's digital mind.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("zane.memory.embeddings")


class EmbeddingError(RuntimeError):
    """Raised when the embedding model fails to load or encode text."""


class EmbeddingBackend:
    """Lazily loads the sentence-transformers model on first use so
    importing this module (or constructing a `SharedBackend`) never pays
    the model-load cost until an embedding is actually requested."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int = 0
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is not installed. Run "
                    "`pip install sentence-transformers` to enable persistent, "
                    "retrieval-augmented memory."
                ) from exc

            try:
                model = SentenceTransformer(self.model_name)
            except Exception as exc:  # noqa: BLE001 - any load failure should degrade, not crash
                raise EmbeddingError(
                    f"Failed to load embedding model {self.model_name!r}: {exc}"
                ) from exc

            self._model = model
            self._dimension = model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return self._dimension

    def embed(self, text: str) -> "np.ndarray":
        """Returns a single L2-normalized embedding vector (float32) for
        `text`. Raises EmbeddingError on any failure; callers should treat
        this as a soft failure and degrade to non-retrieval behavior rather
        than crash the conversation."""
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")

        self._ensure_loaded()
        try:
            import numpy as np

            vector = self._model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
            return np.asarray(vector, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - encode() can raise a variety of backend errors
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

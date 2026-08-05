"""Zane's memory subsystem: a rolling in-context window (`rolling.py`,
unchanged since the original implementation) plus a persistent,
retrieval-augmented layer (`persistent.py`) backed by SQLite, local
sentence-transformer embeddings, and a FAISS vector index.
"""
from zane.memory.persistent import PersistentMemory, PersistentMemoryConfig
from zane.memory.rolling import ConversationMemory

__all__ = ["ConversationMemory", "PersistentMemory", "PersistentMemoryConfig"]

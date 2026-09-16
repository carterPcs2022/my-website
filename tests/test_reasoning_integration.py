from pathlib import Path

import numpy as np

import zane.knowledge_manager as knowledge_manager
from zane.knowledge import KnowledgeVault
from zane.reasoning import ZaneReasoning


class FakeEmbeddings:
    def embed(self, _text):
        return np.zeros(384, dtype=np.float32)


def test_knowledge_vault_loads_raw_text(tmp_path: Path):
    source = tmp_path / "lore.txt"
    source.write_text("Zane is the Master of Ice.\n\nPixal is Zane's companion.", encoding="utf-8")

    vault = KnowledgeVault()
    assert vault.load_text(source) == 2
    result = ZaneReasoning(vault).answer("What element is Zane the master of?")

    assert result.mode == "offline-knowledge"
    assert result.knowledge_used >= 1
    assert "Master of Ice" in result.text


def test_knowledge_manager_exposes_reasoning_in_zanemind_path(tmp_path: Path, monkeypatch):
    source = tmp_path / "raw_lore.txt"
    source.write_text("Zane is the Master of Ice and a Nindroid.", encoding="utf-8")
    monkeypatch.setattr(knowledge_manager, "DEFAULT_RAW_LORE_PATH", str(source))

    manager = knowledge_manager.KnowledgeManager(
        FakeEmbeddings(),
        lore_index_path=str(tmp_path / "lore.index"),
        lore_metadata_path=str(tmp_path / "lore.json"),
    )

    assert manager.reasoning is not None
    results = __import__("asyncio").run(
        manager.query_all_knowledge_sources("Tell me about Zane's element", None, top_k=1)
    )

    assert any(item.startswith("[OFFLINE_REASONING_CONTEXT]") for item in results)
    assert any("Master of Ice" in item for item in results)

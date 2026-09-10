"""Structured Zane <-> P.I.X.A.L. communication primitives.

Messages are data only. This layer does not execute hardware actions; safety
and actuator authorization remain outside the AI message bus.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from typing import Any, Dict, List, Optional


class PixalMessageType(str, Enum):
    OBSERVATION = "observation"
    RECOMMENDATION = "recommendation"
    REQUEST = "request"
    RESPONSE = "response"
    SAFETY_ALERT = "safety_alert"
    CONFIRMATION = "confirmation"
    STATUS = "status"


@dataclass(frozen=True)
class PixalMessage:
    sender: str
    recipient: str
    message_type: PixalMessageType
    content: str
    priority: int = 50
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sender.strip() or not self.recipient.strip():
            raise ValueError("sender and recipient are required")
        if not self.content.strip():
            raise ValueError("message content cannot be empty")
        object.__setattr__(self, "priority", max(0, min(100, int(self.priority))))

    @property
    def is_safety_critical(self) -> bool:
        return self.message_type is PixalMessageType.SAFETY_ALERT or self.priority >= 90


class PixalMessageBus:
    """Bounded in-memory bus with priority-aware retrieval."""

    def __init__(self, *, max_messages: int = 200) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: List[PixalMessage] = []

    def publish(self, message: PixalMessage) -> None:
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            self._messages.sort(key=lambda item: (item.priority, item.created_at))
            self._messages.pop(0)

    def pending_for(self, recipient: str, *, limit: int = 20) -> List[PixalMessage]:
        matches = [m for m in self._messages if m.recipient == recipient]
        matches.sort(key=lambda item: (-item.priority, item.created_at))
        return matches[: max(0, limit)]

    def drain_for(self, recipient: str, *, limit: int = 20) -> List[PixalMessage]:
        selected = self.pending_for(recipient, limit=limit)
        selected_ids = {id(message) for message in selected}
        self._messages = [message for message in self._messages if id(message) not in selected_ids]
        return selected

    def snapshot(self) -> List[PixalMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

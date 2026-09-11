"""Permission policy for outbound email actions."""
from __future__ import annotations

from .models import EmailDraft


class EmailPermissionPolicy:
    """Require an explicit human approval before an email is sent.

    Drafting is safe to automate; sending is an external side effect and is
    therefore intentionally separated from drafting.
    """

    def __init__(self, require_confirmation: bool = True) -> None:
        self.require_confirmation = require_confirmation

    def can_send(self, draft: EmailDraft, *, confirmed: bool = False) -> bool:
        if not draft.to.strip() or not draft.subject.strip() or not draft.body.strip():
            return False
        if self.require_confirmation and not confirmed:
            return False
        return True

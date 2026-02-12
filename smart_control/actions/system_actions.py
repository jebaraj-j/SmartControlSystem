"""Map intent JSON into safe system actions with confirmation rules."""

import logging
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import pyautogui

from smart_control.schemas import IntentResult

LOGGER = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result from command mapping and execution layer."""

    executed: bool
    message: str


class SystemActionMapper:
    """Translate parsed intents into OS-level actions."""

    HIGH_RISK_INTENTS = {"SHUTDOWN"}

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.pending_confirmation_intent: Optional[str] = None
        self._handlers: Dict[str, Callable[[IntentResult], ActionResult]] = {
            "OPEN_APP": self._open_app,
            "VOLUME_UP": self._volume_up,
            "VOLUME_DOWN": self._volume_down,
            "SCROLL_UP": self._scroll_up,
            "SCROLL_DOWN": self._scroll_down,
            "SHUTDOWN": self._shutdown,
        }

    def has_pending_confirmation(self) -> bool:
        """Return whether a high-risk intent is waiting for user confirmation."""
        return self.pending_confirmation_intent is not None

    def confirm_pending(self) -> ActionResult:
        """Execute pending high-risk intent after explicit confirmation."""
        intent = self.pending_confirmation_intent
        if not intent:
            return ActionResult(False, "Nothing to confirm.")
        handler = self._handlers.get(intent)
        if not handler:
            self.pending_confirmation_intent = None
            return ActionResult(False, f"No handler for pending intent: {intent}")
        self.pending_confirmation_intent = None
        return handler(IntentResult(intent=intent, target="", confidence=1.0))

    def cancel_pending(self) -> ActionResult:
        """Cancel pending high-risk intent."""
        if not self.pending_confirmation_intent:
            return ActionResult(False, "Nothing to cancel.")
        canceled = self.pending_confirmation_intent
        self.pending_confirmation_intent = None
        return ActionResult(True, f"Canceled pending intent: {canceled}")

    def execute(self, parsed: IntentResult, confirmation_token: Optional[str] = None) -> ActionResult:
        """Execute mapped action while enforcing confirmation for high-risk intents."""
        intent = parsed.intent.upper()
        if intent in self.HIGH_RISK_INTENTS:
            if self.pending_confirmation_intent == intent and confirmation_token == "CONFIRM":
                self.pending_confirmation_intent = None
                return self._handlers[intent](parsed)

            self.pending_confirmation_intent = intent
            return ActionResult(
                executed=False,
                message="Confirmation required: provide CONFIRM voice text or safe gesture.",
            )

        handler = self._handlers.get(intent)
        if not handler:
            return ActionResult(executed=False, message=f"No handler for intent: {intent}")

        return handler(parsed)

    def _open_app(self, parsed: IntentResult) -> ActionResult:
        target = parsed.target.strip()
        if not target:
            return ActionResult(False, "OPEN_APP missing target")

        if self.dry_run:
            return ActionResult(True, f"[dry-run] Would open app: {target}")

        pyautogui.press("win")
        pyautogui.typewrite(target)
        pyautogui.press("enter")
        return ActionResult(True, f"Opened app: {target}")

    def _volume_up(self, parsed: IntentResult) -> ActionResult:
        steps = self._safe_positive_int(parsed.parameters.get("steps"), default=3, max_value=20)
        if self.dry_run:
            return ActionResult(True, f"[dry-run] Volume up x{steps}")
        for _ in range(steps):
            pyautogui.press("volumeup")
        return ActionResult(True, f"Volume up x{steps}")

    def _volume_down(self, parsed: IntentResult) -> ActionResult:
        steps = self._safe_positive_int(parsed.parameters.get("steps"), default=3, max_value=20)
        if self.dry_run:
            return ActionResult(True, f"[dry-run] Volume down x{steps}")
        for _ in range(steps):
            pyautogui.press("volumedown")
        return ActionResult(True, f"Volume down x{steps}")

    def _scroll_up(self, parsed: IntentResult) -> ActionResult:
        amount = self._safe_positive_int(parsed.parameters.get("amount"), default=600, max_value=5000)
        if self.dry_run:
            return ActionResult(True, f"[dry-run] Scroll up {amount}")
        pyautogui.scroll(abs(amount))
        return ActionResult(True, f"Scrolled up {amount}")

    def _scroll_down(self, parsed: IntentResult) -> ActionResult:
        amount = self._safe_positive_int(parsed.parameters.get("amount"), default=600, max_value=5000)
        if self.dry_run:
            return ActionResult(True, f"[dry-run] Scroll down {amount}")
        pyautogui.scroll(-abs(amount))
        return ActionResult(True, f"Scrolled down {amount}")

    def _shutdown(self, parsed: IntentResult) -> ActionResult:
        if self.dry_run:
            return ActionResult(True, "[dry-run] Would trigger system shutdown")
        subprocess.run(["shutdown", "/s", "/t", "5"], check=False)
        return ActionResult(True, "Shutdown initiated (5 second timer)")

    @staticmethod
    def _safe_positive_int(value: object, default: int, max_value: int) -> int:
        """Parse and clamp numeric parameters from LLM output safely."""
        try:
            parsed = int(value) if value is not None else default
        except (TypeError, ValueError):
            parsed = default
        parsed = max(1, parsed)
        return min(parsed, max_value)

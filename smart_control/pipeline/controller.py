"""Multimodal orchestration logic (gesture + voice + LLM + action mapping)."""

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Optional

from smart_control.actions import ActionResult, SystemActionMapper
from smart_control.llm import Phi3OllamaIntentClassifier
from smart_control.schemas import GestureEvent, MultimodalInput, VoiceEvent

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Return bundle for UI logging and execution feedback."""

    action: ActionResult
    intent: str
    target: str
    confidence: float
    gesture_label: str
    voice_text: str


class SmartControlPipeline:
    """Coordinates sensor outputs with LLM intent parsing and system action execution."""

    def __init__(
        self,
        classifier: Phi3OllamaIntentClassifier,
        action_mapper: SystemActionMapper,
        min_confidence: float = 0.55,
        duplicate_window_sec: float = 2.0,
    ):
        self.classifier = classifier
        self.action_mapper = action_mapper
        self.min_confidence = min_confidence
        self.duplicate_window_sec = duplicate_window_sec
        self.last_gesture: Optional[GestureEvent] = None
        self.last_voice: Optional[VoiceEvent] = None
        self._last_command_key = ""
        self._last_command_ts = 0.0
        self._lock = Lock()

    def update_gesture(self, event: GestureEvent) -> None:
        """Store latest gesture event."""
        with self._lock:
            self.last_gesture = event

    def update_voice(self, event: VoiceEvent) -> PipelineResult:
        """Handle new voice text with latest gesture context and trigger execution."""
        with self._lock:
            self.last_voice = event
            gesture_label = self.last_gesture.label if self.last_gesture else "NO_GESTURE"
            gesture_confidence = self.last_gesture.confidence if self.last_gesture else 0.0

        payload = MultimodalInput(
            gesture_label=gesture_label,
            voice_text=event.text,
            context={
                "gesture_confidence": gesture_confidence,
                "voice_confidence": event.confidence,
            },
        )

        manual = self._handle_manual_confirmation(payload)
        if manual is not None:
            return manual

        parsed = self.classifier.classify(payload)
        if parsed.error:
            return PipelineResult(
                action=ActionResult(False, f"LLM parse error: {parsed.error}"),
                intent="ERROR",
                target="",
                confidence=0.0,
                gesture_label=payload.gesture_label,
                voice_text=payload.voice_text,
            )

        if parsed.confidence < self.min_confidence:
            return PipelineResult(
                action=ActionResult(False, f"Low confidence ({parsed.confidence:.2f}). Ask user to repeat."),
                intent=parsed.intent,
                target=parsed.target,
                confidence=parsed.confidence,
                gesture_label=payload.gesture_label,
                voice_text=payload.voice_text,
            )

        if self._is_duplicate(parsed.intent, parsed.target):
            return PipelineResult(
                action=ActionResult(False, "Duplicate command suppressed."),
                intent=parsed.intent,
                target=parsed.target,
                confidence=parsed.confidence,
                gesture_label=payload.gesture_label,
                voice_text=payload.voice_text,
            )

        confirmation_token = "CONFIRM" if payload.gesture_label == "OPEN_PALM" else None
        result = self.action_mapper.execute(parsed, confirmation_token=confirmation_token)
        LOGGER.info("Intent=%s Target=%s => %s", parsed.intent, parsed.target, result.message)
        return PipelineResult(
            action=result,
            intent=parsed.intent,
            target=parsed.target,
            confidence=parsed.confidence,
            gesture_label=payload.gesture_label,
            voice_text=payload.voice_text,
        )

    def _handle_manual_confirmation(self, payload: MultimodalInput) -> Optional[PipelineResult]:
        text = payload.voice_text.strip().lower()
        if text in {"confirm", "yes", "proceed"}:
            action = self.action_mapper.confirm_pending()
            return PipelineResult(action, "CONFIRM", "", 1.0, payload.gesture_label, payload.voice_text)

        if text in {"cancel", "no", "stop"} or payload.gesture_label == "FIST":
            action = self.action_mapper.cancel_pending()
            return PipelineResult(action, "CANCEL", "", 1.0, payload.gesture_label, payload.voice_text)

        if payload.gesture_label == "OPEN_PALM" and self.action_mapper.has_pending_confirmation():
            action = self.action_mapper.confirm_pending()
            return PipelineResult(action, "CONFIRM", "", 1.0, payload.gesture_label, payload.voice_text)

        return None

    def _is_duplicate(self, intent: str, target: str) -> bool:
        key = f"{intent.upper()}::{target.strip().lower()}"
        now = time.monotonic()
        if key == self._last_command_key and (now - self._last_command_ts) < self.duplicate_window_sec:
            return True
        self._last_command_key = key
        self._last_command_ts = now
        return False

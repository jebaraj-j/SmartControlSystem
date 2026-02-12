"""Shared data models for the multimodal pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class GestureEvent:
    """One normalized gesture observation from the webcam stream."""

    label: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VoiceEvent:
    """One normalized voice transcript event from offline STT."""

    text: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MultimodalInput:
    """Combined gesture + voice state forwarded to the local LLM."""

    gesture_label: str
    voice_text: str
    context: Optional[Dict[str, Any]] = None


@dataclass
class IntentResult:
    """Structured intent extraction output expected from the LLM."""

    intent: str = "UNKNOWN"
    target: str = ""
    confidence: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False
    error: Optional[str] = None
    raw_response: Optional[str] = None

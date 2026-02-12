"""Local Ollama (Phi-3 Mini) intent classification client."""

import json
import logging
import re
from dataclasses import asdict
from typing import Any, Dict

import requests

from smart_control.schemas import IntentResult, MultimodalInput

LOGGER = logging.getLogger(__name__)


class LLMOutputError(RuntimeError):
    """Raised when the LLM output cannot be converted to valid intent JSON."""


class Phi3OllamaIntentClassifier:
    """Classify combined gesture + voice input via local Ollama-hosted Phi-3 Mini."""

    def __init__(self, model: str = "phi3:mini", endpoint: str = "http://127.0.0.1:11434/api/generate", timeout: int = 20):
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout

    def classify(self, payload: MultimodalInput) -> IntentResult:
        """Send multimodal payload to LLM and parse structured response safely."""
        prompt = self._build_prompt(payload)
        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
            parsed = self._extract_json(raw_text)
        except requests.RequestException as exc:
            LOGGER.error("Ollama request failed: %s", exc)
            return IntentResult(error=f"LLM unavailable: {exc}")
        except LLMOutputError as exc:
            LOGGER.warning("Failed to parse LLM output")
            return IntentResult(error=str(exc), raw_response=raw_text)

        return IntentResult(
            intent=str(parsed.get("intent", "UNKNOWN")).upper(),
            target=str(parsed.get("target", "")),
            confidence=float(parsed.get("confidence", 0.0)),
            parameters=parsed.get("parameters", {}) if isinstance(parsed.get("parameters", {}), dict) else {},
            requires_confirmation=bool(parsed.get("requires_confirmation", False)),
            raw_response=raw_text,
        )

    def _build_prompt(self, payload: MultimodalInput) -> str:
        """Create a deterministic prompt that asks for JSON-only output."""
        safe_payload = asdict(payload)
        return (
            "You are an intent parser for a Windows smart-control system. "
            "Return only valid JSON with keys: intent, target, confidence, parameters, requires_confirmation.\n"
            "Allowed intents: OPEN_APP, CLOSE_APP, SHUTDOWN, VOLUME_UP, VOLUME_DOWN, SCROLL_UP, SCROLL_DOWN, UNKNOWN.\n"
            "Confidence must be 0.0 to 1.0.\n"
            f"Input payload: {json.dumps(safe_payload)}"
        )

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Extract JSON object from plain or markdown-wrapped model output."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            cleaned = re.sub(r"```$", "", cleaned).strip()

        for candidate in (cleaned, self._first_json_like_block(cleaned)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        raise LLMOutputError("LLM output was not valid JSON.")

    @staticmethod
    def _first_json_like_block(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0).strip() if match else ""

"""Template script: call local Phi-3 Mini via Ollama for intent JSON extraction."""

from smart_control.llm import Phi3OllamaIntentClassifier
from smart_control.schemas import MultimodalInput


def main() -> None:
    classifier = Phi3OllamaIntentClassifier(model="phi3:mini")
    payload = MultimodalInput(
        gesture_label="TWO_FINGERS",
        voice_text="open chrome tab",
        context={"app": "browser"},
    )
    result = classifier.classify(payload)
    print(result)


if __name__ == "__main__":
    main()

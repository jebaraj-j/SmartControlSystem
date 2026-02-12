"""Template script: offline speech with Vosk (streaming mic)."""

import time

from smart_control.logging_config import configure_logging
from smart_control.speech import MicrophoneError, VoskSpeechConfig, VoskSpeechRecognizer


def main() -> None:
    configure_logging()
    recognizer = VoskSpeechRecognizer(
        VoskSpeechConfig(model_path="voice/vosk-model-small-en-us-0.15")
    )

    def on_event(event):
        print(f"Voice: {event.text} (conf={event.confidence:.2f})")

    def on_error(exc: Exception):
        print(f"Speech error: {exc}")

    try:
        recognizer.start(on_event=on_event, on_error=on_error)
        print("Listening... press Ctrl+C to stop")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    except (FileNotFoundError, MicrophoneError) as exc:
        print(f"Startup error: {exc}")
    finally:
        recognizer.stop()


if __name__ == "__main__":
    main()

"""Template script: end-to-end multimodal sample using cvzone + Vosk + Phi-3 + mapper."""

import logging
import threading
import time

import cv2

from smart_control.actions import SystemActionMapper
from smart_control.gesture import CvzoneGestureDetector, WebcamError
from smart_control.llm import Phi3OllamaIntentClassifier
from smart_control.logging_config import configure_logging
from smart_control.pipeline import SmartControlPipeline
from smart_control.speech import MicrophoneError, VoskSpeechConfig, VoskSpeechRecognizer

LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging(logging.INFO)

    detector = CvzoneGestureDetector(camera_index=0)
    recognizer = VoskSpeechRecognizer(VoskSpeechConfig(model_path="voice/vosk-model-small-en-us-0.15"))
    classifier = Phi3OllamaIntentClassifier(model="phi3:mini")
    mapper = SystemActionMapper(dry_run=True)
    pipeline = SmartControlPipeline(classifier=classifier, action_mapper=mapper)

    stop_event = threading.Event()

    def on_voice(event):
        result = pipeline.update_voice(event)
        LOGGER.info("Action result: %s", result.message)

    def on_voice_error(exc: Exception):
        LOGGER.error("Voice pipeline error: %s", exc)

    try:
        detector.open()
        recognizer.start(on_event=on_voice, on_error=on_voice_error)
    except (WebcamError, FileNotFoundError, MicrophoneError) as exc:
        LOGGER.error("Startup failure: %s", exc)
        return

    LOGGER.info("Running demo. Press 'q' in video window to quit.")

    try:
        while not stop_event.is_set():
            frame, event = detector.read()
            if frame is None:
                time.sleep(0.01)
                continue

            if event is not None:
                pipeline.update_gesture(event)

            cv2.imshow("Smart Control Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
    finally:
        recognizer.stop()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

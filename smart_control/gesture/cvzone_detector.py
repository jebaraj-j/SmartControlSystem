"""cvzone-based hand gesture detection template."""

import logging
from typing import Optional, Tuple

import cv2
import mediapipe as mp

# cvzone expects `mediapipe.solutions`, but recent mediapipe releases
# expose `solutions` only under `mediapipe.python`.
if not hasattr(mp, "solutions"):
    try:
        from mediapipe.python import solutions as mp_solutions
    except Exception as exc:
        raise ImportError(
            "cvzone requires mediapipe with `solutions` API. "
            "Install a compatible version such as mediapipe==0.10.10."
        ) from exc
    else:
        mp.solutions = mp_solutions

from cvzone.HandTrackingModule import HandDetector

from smart_control.schemas import GestureEvent

LOGGER = logging.getLogger(__name__)


class WebcamError(RuntimeError):
    """Raised when webcam cannot be opened or frame capture fails repeatedly."""


class CvzoneGestureDetector:
    """Capture webcam frames and convert cvzone hand state into clean gesture labels."""

    def __init__(self, camera_index: int = 0, detection_confidence: float = 0.8):
        self.camera_index = camera_index
        self.detector = HandDetector(maxHands=1, detectionCon=detection_confidence)
        self.cap: Optional[cv2.VideoCapture] = None
        self._read_failures = 0

    def open(self) -> None:
        """Open webcam and validate availability."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap or not self.cap.isOpened():
            raise WebcamError(
                f"Webcam index {self.camera_index} is unavailable. "
                "Check device connection and camera permissions."
            )
        LOGGER.info("Webcam opened on index %s", self.camera_index)

    def close(self) -> None:
        """Release webcam resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            LOGGER.info("Webcam released")

    def read(self) -> Tuple[Optional[object], Optional[GestureEvent]]:
        """Read one frame and return a normalized `GestureEvent` when possible."""
        if self.cap is None:
            raise WebcamError("Webcam is not open. Call open() first.")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._read_failures += 1
            if self._read_failures >= 10:
                raise WebcamError("Failed to read frames from webcam 10 times in a row.")
            return None, None

        self._read_failures = 0
        hands, frame = self.detector.findHands(frame, draw=True)
        if not hands:
            return frame, None

        fingers = self.detector.fingersUp(hands[0])
        label, score = self._map_fingers_to_label(fingers)
        event = GestureEvent(label=label, confidence=score)
        cv2.putText(frame, f"{label} ({score:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame, event

    @staticmethod
    def _map_fingers_to_label(fingers: list) -> Tuple[str, float]:
        """Map finger-state vector to project-specific gesture labels."""
        total = sum(fingers)
        if total == 0:
            return "FIST", 0.95
        if total == 5:
            return "OPEN_PALM", 0.95
        if fingers[:2] == [0, 1] and fingers[2:] == [1, 0, 0]:
            return "TWO_FINGERS", 0.85
        if fingers == [0, 1, 0, 0, 0]:
            return "ONE_FINGER", 0.80
        return "UNKNOWN_GESTURE", 0.40

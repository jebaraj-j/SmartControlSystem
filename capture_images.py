"""
capture_images.py — Multi-hand image collector
"""

import cv2
import mediapipe as mp
import os
import numpy as np
import sys
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import GESTURE_CLASSES, GESTURE_INSTRUCTIONS

DATASET_ROOT = "dataset"
IMAGE_SIZE = (224, 224)


def crop_hand(frame, landmarks):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))

    pad_x = int((x1 - x0) * 0.2)
    pad_y = int((y1 - y0) * 0.2)

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    roi = frame[y0:y1, x0:x1]
    return cv2.resize(roi, IMAGE_SIZE) if roi.size else None


class ManualCollector:

    def __init__(self):

        # ✅ CAMERA FIX
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("❌ Camera error")
            sys.exit(1)

        assert os.path.exists("hand_landmarker.task"), "❌ model missing"

        BaseOptions = python.BaseOptions
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def _dir(self, name):
        path = os.path.join(DATASET_ROOT, name)
        os.makedirs(path, exist_ok=True)
        return path

    def _next_index(self, name):
        files = [f for f in os.listdir(self._dir(name)) if f.endswith(".jpg")]
        return len(files)

    def _collect(self, name):
        idx = self._next_index(name)
        print(f"\nCollecting: {name}")
        print(GESTURE_INSTRUCTIONS.get(name, ""))

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(time.time() * 1000)
            result = self.landmarker.detect_for_video(mp_img, timestamp)

            hand_count = len(result.hand_landmarks) if result.hand_landmarks else 0

            cv2.putText(frame, f"Hands: {hand_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            cv2.imshow("Capture", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if name == "Both_Shutdown" and hand_count != 2:
                    print("⚠ Show TWO hands")
                    continue
                if hand_count == 0:
                    continue

                crop = crop_hand(frame, result.hand_landmarks[0])
                if crop is None:
                    continue

                path = os.path.join(self._dir(name), f"{name}_{idx:04d}.jpg")
                cv2.imwrite(path, crop)
                idx += 1
                print("✓ Saved", path)

            elif key == ord("q"):
                return

    def run(self):
        for i, g in enumerate(GESTURE_CLASSES):
            print(f"{i+1}. {g}")

        while True:
            ch = input("Select gesture (Q quit): ").lower()
            if ch == "q":
                break
            if ch.isdigit():
                self._collect(GESTURE_CLASSES[int(ch)-1])

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ManualCollector().run()

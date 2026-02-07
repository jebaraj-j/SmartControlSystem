"""
capture_images.py — Manual one-press-at-a-time image collector.
Uses MediaPipe Tasks API (compatible with mediapipe 0.10.32)

Controls (camera window):
SPACE  → save image
Q      → next gesture
ESC    → quit
"""

import cv2
import mediapipe as mp
import os
import sys

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import GESTURE_CLASSES, GESTURE_INSTRUCTIONS

DATASET_ROOT = "dataset"
IMAGE_SIZE = (224, 224)


# ─────────────────────────────────────────────
# Crop hand using landmarks (Tasks API)
# ─────────────────────────────────────────────
def crop_hand(frame, landmarks):
    h, w, _ = frame.shape

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))

    px, py = int((x1 - x0) * 0.2), int((y1 - y0) * 0.2)
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(w, x1 + px)
    y1 = min(h, y1 + py)

    region = frame[y0:y1, x0:x1]
    return cv2.resize(region, IMAGE_SIZE) if region.size else None


# ─────────────────────────────────────────────
# Manual Collector
# ─────────────────────────────────────────────
class ManualCollector:
    def __init__(self):
        # Camera (Windows-safe)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("❌ Camera could not be opened")
            sys.exit(1)

        # MediaPipe Tasks Hand Landmarker
        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    # ─────────────────────────────────────────
    @staticmethod
    def _dir(name):
        path = os.path.join(DATASET_ROOT, name)
        os.makedirs(path, exist_ok=True)
        return path

    def _count(self, name):
        return len([f for f in os.listdir(self._dir(name)) if f.endswith(".jpg")])

    def _next_idx(self, name):
        files = [f for f in os.listdir(self._dir(name)) if f.endswith(".jpg")]
        if not files:
            return 0
        nums = [int(f.split("_")[-1].replace(".jpg", "")) for f in files]
        return max(nums) + 1 if nums else 0

    # ─────────────────────────────────────────
    def _draw_ui(self, frame, name, total, hand_ok, flash):
        h, w, _ = frame.shape

        cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
        cv2.putText(frame, f"Gesture: {name}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {total}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        status = "HAND OK" if hand_ok else "NO HAND"
        color = (0, 255, 0) if hand_ok else (0, 0, 255)
        cv2.putText(frame, status, (w - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if flash:
            cv2.rectangle(frame, (5, 5), (w - 5, h - 5), (0, 0, 255), 5)

        cv2.rectangle(frame, (0, h - 50), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, GESTURE_INSTRUCTIONS[name], (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # ─────────────────────────────────────────
    def _collect(self, name):
        existing = self._count(name)
        idx = self._next_idx(name)
        added = 0
        flash = 0

        print(f"\n── {name} ── ({existing} images)")
        print(GESTURE_INSTRUCTIONS[name])

        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_ok = bool(result.hand_landmarks)
            landmarks = result.hand_landmarks[0] if hand_ok else None

            if hand_ok:
                for lm in landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            flash = max(0, flash - 1)
            self._draw_ui(frame, name, existing + added, hand_ok, flash > 0)
            cv2.imshow("Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if name != "No_Hand" and not hand_ok:
                    print("⚠ Show hand first")
                    continue

                img = crop_hand(frame, landmarks) if name != "No_Hand" else cv2.resize(frame, IMAGE_SIZE)
                if img is None:
                    continue

                path = os.path.join(self._dir(name), f"{name}_{idx:04d}.jpg")
                cv2.imwrite(path, img)
                idx += 1
                added += 1
                flash = 6
                print(f"✓ Saved {path}")

            elif key == ord("q"):
                return added

            elif key == 27:
                return -1

    # ─────────────────────────────────────────
    def run(self):
        print("\nMANUAL GESTURE IMAGE COLLECTOR\n")

        for i, g in enumerate(GESTURE_CLASSES):
            print(f"{i + 1}. {g}")

        while True:
            ch = input("\nChoice (1-11 / A / Q): ").strip().lower()
            if ch == "q":
                break
            elif ch == "a":
                for g in GESTURE_CLASSES:
                    if self._collect(g) == -1:
                        break
            elif ch.isdigit() and 1 <= int(ch) <= len(GESTURE_CLASSES):
                if self._collect(GESTURE_CLASSES[int(ch) - 1]) == -1:
                    break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\nDone.")


if __name__ == "__main__":
    ManualCollector().run()

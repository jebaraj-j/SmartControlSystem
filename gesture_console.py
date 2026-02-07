"""
gesture_console.py  —  Console-only identification system (FIXED).
Uses MediaPipe Tasks HandLandmarker + trained CNN.
"""

import cv2
import mediapipe as mp
import numpy as np
import os, sys, json, time

# ─────────────────────────────────────────────
# project imports
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import GESTURE_CLASSES, GESTURE_ACTION
from capture_images import crop_hand

# ─────────────────────────────────────────────
# config
# ─────────────────────────────────────────────
MODEL_PATH = "gesture_model.h5"
CLASS_NAMES_FILE = "class_names.json"
IMAGE_SIZE = (224, 224)
PRINT_INTERVAL = 0.5

# ─────────────────────────────────────────────
# console colors
# ─────────────────────────────────────────────
class C:
    G = "\033[92m"
    Y = "\033[93m"
    B = "\033[94m"
    M = "\033[95m"
    C = "\033[96m"
    R = "\033[91m"
    BD = "\033[1m"
    _ = "\033[0m"


def banner():
    print(C.C + C.BD + """
╔════════════════════════════════════════════════════════════╗
║          SMART CONTROL SYSTEM — CONSOLE MODE               ║
║                                                            ║
║   Gesture → CNN classifies hand → prints gesture + action  ║
║                                                            ║
║   ⚠  IDENTIFY ONLY — nothing is executed                   ║
╚════════════════════════════════════════════════════════════╝
""" + C._)


# ═════════════════════════════════════════════════════════════
# GESTURE MODE
# ═════════════════════════════════════════════════════════════
class GestureMode:
    def __init__(self):
        self.model = None
        self.class_names = None
        self.landmarker = None

    # ─────────────────────────────────────────
    # load CNN
    # ─────────────────────────────────────────
    def load_model(self):
        import tensorflow as tf

        if not os.path.exists(MODEL_PATH):
            print(f"{C.R}❌ gesture_model.h5 not found{C._}")
            return False

        if not os.path.exists(CLASS_NAMES_FILE):
            print(f"{C.R}❌ class_names.json not found{C._}")
            return False

        print(f"{C.C}🤖 Loading CNN model…{C._}")
        self.model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_NAMES_FILE) as f:
            self.class_names = json.load(f)

        print(f"{C.G}✓ Model ready{C._}")
        return True

    # ─────────────────────────────────────────
    # load MediaPipe Tasks
    # ─────────────────────────────────────────
    def load_landmarker(self):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print(f"{C.R}❌ hand_landmarker.task not found{C._}")
            return False

        BaseOptions = python.BaseOptions
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        return True

    # ─────────────────────────────────────────
    # CNN prediction
    # ─────────────────────────────────────────
    def predict(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = rgb.astype("float32") / 255.0
        x = np.expand_dims(x, 0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])

    # ─────────────────────────────────────────
    # print detection
    # ─────────────────────────────────────────
    def print_detection(self, name, conf):
        print(f"\n{C.M}{C.BD}┌─────────────────────────────────────────┐{C._}")

        if name == "Cursor":
            print(f"{C.G}{C.BD}│ Gesture : {name:<25} {conf*100:5.1f}% │{C._}")
        elif name == "No_Hand":
            print(f"{C.Y}{C.BD}│ No hand detected                       │{C._}")
        else:
            action = GESTURE_ACTION.get(name, "—")
            print(f"{C.G}{C.BD}│ Gesture : {name:<25} {conf*100:5.1f}% │{C._}")
            print(f"{C.B}{C.BD}│ Action  : {action:<25}       │{C._}")

        print(f"{C.M}{C.BD}└─────────────────────────────────────────┘{C._}")

    # ─────────────────────────────────────────
    # main loop
    # ─────────────────────────────────────────
    def run(self):
        if not self.load_model():
            return
        if not self.load_landmarker():
            return

        print(f"{C.C}📹 Opening camera…{C._}")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(f"{C.R}❌ Camera not available{C._}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"{C.G}✓ Camera started{C._}")
        print(f"{C.Y}Show your hand — Ctrl+C to stop{C._}\n")

        last_name = None
        last_time = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(time.time() * 1000)
                result = self.landmarker.detect_for_video(mp_image, timestamp)

                if result.hand_landmarks:
                    for lms in result.hand_landmarks:
                        hand_img = crop_hand(frame, lms)
                        if hand_img is None:
                            continue

                        name, conf = self.predict(hand_img)

                        now = time.time()
                        if name != last_name or (now - last_time) > PRINT_INTERVAL:
                            self.print_detection(name, conf)
                            last_name = name
                            last_time = now
                else:
                    if last_name != "No_Hand":
                        print(f"{C.Y}[no hand]{C._}")
                        last_name = "No_Hand"

                time.sleep(0.03)

        except KeyboardInterrupt:
            print(f"\n{C.C}Stopped{C._}")
        finally:
            cap.release()


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════
def main():
    banner()
    print(f"{C.G}1. Gesture recognition{C._}\n")

    ch = input("pick (1): ").strip()
    if ch == "1":
        GestureMode().run()


if __name__ == "__main__":
    main()

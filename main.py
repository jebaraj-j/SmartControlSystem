import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import mediapipe as mp
import pyautogui
import sounddevice as sd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vosk import KaldiRecognizer, Model

from config import get_config


@dataclass
class VoiceCommandResult:
    text: str
    intent: str
    action: str
    status: str


class VoiceCommandProcessor:
    def __init__(self):
        self.pending_confirmation: Optional[str] = None

    def process_text(self, text: str) -> VoiceCommandResult:
        raw = text.strip()
        lowered = raw.lower()

        if lowered in {"confirm", "yes", "proceed"}:
            return self._confirm(raw)
        if lowered in {"cancel", "no", "stop"}:
            return self._cancel(raw)

        if lowered.startswith("open "):
            target = lowered.replace("open ", "", 1).strip()
            return self._open_app(raw, target)
        if lowered in {"close window", "close app", "close"}:
            pyautogui.hotkey("alt", "f4")
            return VoiceCommandResult(raw, "CLOSE_WINDOW", "ALT+F4", "Closed active window.")
        if lowered in {"switch window", "next window"}:
            pyautogui.hotkey("alt", "tab")
            return VoiceCommandResult(raw, "SWITCH_WINDOW", "ALT+TAB", "Switched window.")
        if lowered in {"minimize", "minimize window"}:
            pyautogui.hotkey("alt", "space")
            pyautogui.press("n")
            return VoiceCommandResult(raw, "MINIMIZE", "MINIMIZE", "Minimized active window.")
        if lowered in {"maximize", "maximize window"}:
            pyautogui.hotkey("alt", "space")
            pyautogui.press("x")
            return VoiceCommandResult(raw, "MAXIMIZE", "MAXIMIZE", "Maximized active window.")
        if lowered in {"scroll up"}:
            pyautogui.scroll(600)
            return VoiceCommandResult(raw, "SCROLL_UP", "SCROLL_UP", "Scrolled up.")
        if lowered in {"scroll down"}:
            pyautogui.scroll(-600)
            return VoiceCommandResult(raw, "SCROLL_DOWN", "SCROLL_DOWN", "Scrolled down.")
        if lowered == "shutdown":
            self.pending_confirmation = "shutdown"
            return VoiceCommandResult(raw, "SHUTDOWN", "PENDING_SHUTDOWN", "Say confirm to shutdown.")
        if lowered == "sleep":
            self.pending_confirmation = "sleep"
            return VoiceCommandResult(raw, "SLEEP", "PENDING_SLEEP", "Say confirm to sleep.")

        return VoiceCommandResult(raw, "UNKNOWN", "NONE", "Command not recognized.")

    def _open_app(self, raw: str, target: str) -> VoiceCommandResult:
        if not target:
            return VoiceCommandResult(raw, "OPEN_APP", "NONE", "Open what?")

        folder_path = self._known_folder_path(target)
        if folder_path:
            os.startfile(folder_path)
            return VoiceCommandResult(raw, "OPEN_FOLDER", "OPEN_FOLDER", f"Opened folder: {folder_path}")

        pyautogui.press("win")
        time.sleep(0.1)
        pyautogui.typewrite(target)
        time.sleep(0.1)
        pyautogui.press("enter")
        return VoiceCommandResult(raw, "OPEN_APP", "OPEN_APP", f"Opening app: {target}")

    @staticmethod
    def _known_folder_path(target: str) -> Optional[str]:
        key = re.sub(r"[^a-z]+", "", target.lower())
        mapping = {
            "downloads": str(Path.home() / "Downloads"),
            "documents": str(Path.home() / "Documents"),
            "pictures": str(Path.home() / "Pictures"),
            "videos": str(Path.home() / "Videos"),
            "desktop": str(Path.home() / "Desktop"),
        }
        return mapping.get(key)

    def _confirm(self, raw: str) -> VoiceCommandResult:
        if self.pending_confirmation == "shutdown":
            subprocess.Popen(["shutdown", "/s", "/t", "5"], shell=False)
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CONFIRM", "SHUTDOWN", "Shutdown initiated in 5 seconds.")
        if self.pending_confirmation == "sleep":
            subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], shell=False)
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CONFIRM", "SLEEP", "Sleep initiated.")
        return VoiceCommandResult(raw, "CONFIRM", "NONE", "Nothing pending.")

    def _cancel(self, raw: str) -> VoiceCommandResult:
        if self.pending_confirmation == "shutdown":
            subprocess.Popen(["shutdown", "/a"], shell=False)
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CANCEL", "CANCEL_SHUTDOWN", "Shutdown canceled.")
        if self.pending_confirmation == "sleep":
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CANCEL", "CANCEL_SLEEP", "Sleep canceled.")
        return VoiceCommandResult(raw, "CANCEL", "NONE", "Nothing pending.")


class VoskStreamRecognizer:
    def __init__(self, model_path: Path, sample_rate: int = 16000):
        self.model = Model(str(model_path))
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)
        self.sample_rate = sample_rate
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._run_flag = False
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _resolve_device() -> int:
        default_device = sd.default.device[0] if sd.default.device else None
        if isinstance(default_device, int) and default_device >= 0:
            info = sd.query_devices(default_device)
            if info.get("max_input_channels", 0) > 0:
                return default_device
        for idx, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                return idx
        raise RuntimeError("No input microphone found.")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        self._queue.put(indata.tobytes())

    def start(
        self,
        on_text: Callable[[str], None],
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._run_flag:
            return
        self._run_flag = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(on_text, on_error),
            daemon=True,
        )
        self._thread.start()

    def _run_loop(
        self,
        on_text: Callable[[str], None],
        on_error: Optional[Callable[[str], None]],
    ) -> None:
        try:
            device = self._resolve_device()
            with sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.sample_rate,
                dtype="int16",
                blocksize=4000,
                callback=self._audio_callback,
            ):
                while self._run_flag:
                    try:
                        data = self._queue.get(timeout=0.25)
                    except queue.Empty:
                        continue

                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            on_text(text)
        except Exception as exc:
            if on_error:
                on_error(f"Mic error: {exc}")
        finally:
            self._run_flag = False

    def stop(self) -> None:
        self._run_flag = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None


class VirtualMouse:
    def __init__(self):
        pyautogui.FAILSAFE = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.smooth_factor = 6

    def move_cursor(self, x_norm: float, y_norm: float) -> None:
        x = (1.0 - x_norm) * self.screen_w
        y = y_norm * self.screen_h
        if self.prev_x is not None and self.prev_y is not None:
            x = self.prev_x + (x - self.prev_x) / self.smooth_factor
            y = self.prev_y + (y - self.prev_y) / self.smooth_factor
        pyautogui.moveTo(x, y)
        self.prev_x, self.prev_y = x, y


class GestureEngine:
    def __init__(self, hand_landmarker_path: Path, camera_index: int = 0):
        self.hand_landmarker_path = hand_landmarker_path
        self.camera_index = camera_index
        self.cap = None
        self.landmarker = None
        self.mouse = VirtualMouse()
        self.last_action_time = 0.0
        self.action_cooldown = 0.35

    def load(self) -> None:
        if not self.hand_landmarker_path.exists():
            raise FileNotFoundError(f"Hand landmarker not found: {self.hand_landmarker_path}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.hand_landmarker_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera not available at index {self.camera_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_gesture(self) -> tuple[str, float]:
        ok, frame = self.cap.read()
        if not ok:
            return "No_Hand", 0.0

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.time() * 1000)
        result = self.landmarker.detect_for_video(mp_image, ts)

        if not result.hand_landmarks:
            return "No_Hand", 1.0

        hand_landmarks = result.hand_landmarks[0]
        handedness = "Right"
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name

        fingers = self._fingers_up(hand_landmarks, handedness)
        label, confidence = self._map_gesture(fingers)
        self._execute_from_gesture(label, hand_landmarks)
        return label, confidence

    def _fingers_up(self, landmarks, handedness: str) -> list[int]:
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        out = [0, 0, 0, 0, 0]

        thumb_tip = landmarks[tips[0]]
        thumb_joint = landmarks[pips[0]]
        if handedness.lower() == "right":
            out[0] = 1 if thumb_tip.x < thumb_joint.x else 0
        else:
            out[0] = 1 if thumb_tip.x > thumb_joint.x else 0

        for i in range(1, 5):
            out[i] = 1 if landmarks[tips[i]].y < landmarks[pips[i]].y else 0
        return out

    @staticmethod
    def _map_gesture(fingers: list[int]) -> tuple[str, float]:
        if fingers == [0, 1, 0, 0, 0]:
            return "R_Cursor", 0.90
        if fingers == [0, 1, 1, 0, 0]:
            return "R_Scroll_Up", 0.85
        if fingers == [0, 1, 1, 1, 1]:
            return "R_Scroll_Down", 0.80
        if sum(fingers) == 0:
            return "R_Closed_Fist", 0.95
        if sum(fingers) == 5:
            return "R_Open_Palm", 0.95
        return "Unknown", 0.50

    def _execute_from_gesture(self, label: str, landmarks) -> None:
        now = time.monotonic()

        if label == "R_Cursor":
            index_tip = landmarks[8]
            self.mouse.move_cursor(index_tip.x, index_tip.y)
            return

        if now - self.last_action_time < self.action_cooldown:
            return

        if label == "R_Open_Palm":
            pyautogui.click()
            self.last_action_time = now
        elif label == "R_Closed_Fist":
            pyautogui.rightClick()
            self.last_action_time = now
        elif label == "R_Scroll_Up":
            pyautogui.scroll(250)
            self.last_action_time = now
        elif label == "R_Scroll_Down":
            pyautogui.scroll(-250)
            self.last_action_time = now


def run_voice_loop(stop_event: threading.Event, recognizer: VoskStreamRecognizer, processor: VoiceCommandProcessor) -> None:
    def on_text(text: str) -> None:
        result = processor.process_text(text)
        print(
            f"[VOICE] \"{result.text}\" | intent={result.intent} | "
            f"action={result.action} | status={result.status}"
        )

    def on_error(message: str) -> None:
        print(f"[VOICE] {message}")

    recognizer.start(on_text=on_text, on_error=on_error)
    print("[VOICE] listening")

    while not stop_event.is_set():
        time.sleep(0.2)

    recognizer.stop()
    print("[VOICE] stopped")


def run_gesture_loop(stop_event: threading.Event, engine: GestureEngine, print_interval_sec: float) -> None:
    try:
        engine.load()
        print("[GESTURE] camera active")
    except Exception as exc:
        print(f"[GESTURE] startup failed: {exc}")
        return

    last_label = ""
    last_print = 0.0
    try:
        while not stop_event.is_set():
            label, conf = engine.read_gesture()
            now = time.monotonic()
            if label != last_label or (now - last_print) >= print_interval_sec:
                print(f"[GESTURE] {label} ({conf:.2f})")
                last_label = label
                last_print = now
            time.sleep(0.02)
    except Exception as exc:
        print(f"[GESTURE] runtime error: {exc}")
    finally:
        engine.close()
        print("[GESTURE] camera stopped")


def main() -> None:
    cfg = get_config()
    stop_event = threading.Event()

    voice_processor = VoiceCommandProcessor()
    voice_recognizer = VoskStreamRecognizer(model_path=cfg.vosk_model_path)
    gesture_engine = GestureEngine(
        hand_landmarker_path=cfg.hand_landmarker_path,
        camera_index=cfg.camera_index,
    )

    voice_thread = threading.Thread(
        target=run_voice_loop,
        args=(stop_event, voice_recognizer, voice_processor),
        daemon=True,
    )
    gesture_thread = threading.Thread(
        target=run_gesture_loop,
        args=(stop_event, gesture_engine, cfg.gesture_print_interval_sec),
        daemon=True,
    )

    print("Smart Control System - Single File Runtime")
    print(f"Vosk Model: {cfg.vosk_model_path}")
    print(f"Hand Landmarker: {cfg.hand_landmarker_path}")
    print("Press Ctrl+C to stop.")

    voice_thread.start()
    gesture_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        stop_event.set()
        voice_thread.join(timeout=2.0)
        gesture_thread.join(timeout=2.0)
        print("Stopped.")


if __name__ == "__main__":
    main()

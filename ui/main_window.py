import sys
import os
import time
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame,
    QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from gesture.detector import HandDetector
from gesture.virtual_mouse import VirtualMouse
from voice.recognizer import VoskConfig, VoskStreamRecognizer
from voice.intent_extractor import VoiceCommandProcessor


# ─────────────────────────────────────────
# VOICE THREAD
# ─────────────────────────────────────────
class VoiceThread(QThread):
    command_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    mic_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.processor = VoiceCommandProcessor()
        self.recognizer = None

    def run(self):
        self._run_flag = True
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_model = os.path.join(base_dir, "voice", "vosk-model-small-en-us-0.15")
        model_path = os.environ.get("VOSK_MODEL_PATH", default_model)
        device_env = os.environ.get("VOSK_DEVICE")
        device = int(device_env) if device_env and device_env.isdigit() else device_env

        if not os.path.isdir(model_path):
            self.status_signal.emit(f"Vosk model not found: {model_path}")
            self.mic_signal.emit(False)
            return

        try:
            self.recognizer = VoskStreamRecognizer(
                VoskConfig(
                    model_path=model_path,
                    device=device,
                )
            )
        except Exception as e:
            self.status_signal.emit(str(e))
            self.mic_signal.emit(False)
            return

        self.mic_signal.emit(True)

        def on_text(text):
            if not self._run_flag:
                return
            self.command_signal.emit(text)
            cmd = self.processor.parse_command(text)
            status = self.processor.execute(cmd)
            self.status_signal.emit(status)

        def on_error(msg):
            self.status_signal.emit(msg)
            self.mic_signal.emit(False)
            self._run_flag = False

        self.recognizer.start(on_text, on_error)
        self.status_signal.emit("Mic listening...")

        while self._run_flag:
            self.msleep(200)

        self.recognizer.stop()
        self.mic_signal.emit(False)

    def stop(self):
        self._run_flag = False
        self.wait()


# ─────────────────────────────────────────
# CAMERA THREAD
# ─────────────────────────────────────────
class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_update_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = HandDetector(max_hands=2)
        self.vm = VirtualMouse()
        self.start_time = time.perf_counter()

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qimg.copy())

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# ─────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Control System – Gesture & Voice")
        self.resize(1200, 800)

        self.system_running = False
        self.voice_running = False

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:black; color:white;")
        self.video_label.setMinimumHeight(400)

        self.voice_status = QLabel("🎤 Mic: Inactive")
        self.voice_status.setStyleSheet("font-size:14pt; padding:10px;")

        self.voice_button = QPushButton("🎤 Start Voice")
        self.voice_button.setMinimumHeight(45)
        self.voice_button.clicked.connect(self.toggle_voice)

        layout.addWidget(self.video_label)
        layout.addWidget(self.voice_status)
        layout.addWidget(self.voice_button)

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)

        self.voice_thread = VoiceThread()
        self.voice_thread.command_signal.connect(self.update_voice_command)
        self.voice_thread.status_signal.connect(self.update_voice_status)
        self.voice_thread.mic_signal.connect(self.update_mic_status)

    def toggle_voice(self):
        if not self.voice_running:
            self.voice_thread.start()
            self.voice_button.setText("🛑 Stop Voice")
            self.voice_running = True
        else:
            self.voice_thread.stop()
            self.voice_button.setText("🎤 Start Voice")
            self.voice_running = False

    def update_image(self, img):
        self.video_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def update_voice_command(self, text):
        self.voice_status.setText(f"🗣 {text}")

    def update_voice_status(self, text):
        self.voice_status.setText(f"ℹ {text}")

    def update_mic_status(self, active):
        self.voice_status.setText(
            "🎤 Mic: Listening" if active else "🎤 Mic: Inactive"
        )

    def closeEvent(self, event):
        if self.voice_running:
            self.voice_thread.stop()
        event.accept()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

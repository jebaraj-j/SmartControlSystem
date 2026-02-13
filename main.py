# main.py
import sys
import json
import os
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

# Check NumPy version before other imports
try:
    import numpy as np
    if int(np.__version__.split('.')[0]) >= 2:
        print("NumPy 2.x detected. Installing NumPy 1.x...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2", "--force-reinstall"])
        print("Please restart the application.")
        sys.exit(1)
except:
    pass

# PyQt5 imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import QSound

# Computer vision imports
import cv2
import mediapipe as mp
import pyautogui
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import joblib

# Set DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

# ==================== Configuration ====================

@dataclass
class Config:
    vosk_model_path: Path = Path("models/vosk-model-small-en-us-0.15")
    hand_landmarker_path: Path = Path("models/hand_landmarker.task")
    gesture_model_path: Path = Path("gesture_model.pkl")
    camera_index: int = 0
    gesture_print_interval_sec: float = 0.5

def get_config() -> Config:
    return Config()

# ==================== Voice Command Processor ====================

@dataclass
class VoiceCommandResult:
    text: str
    intent: str
    action: str
    status: str

class VoiceCommandProcessor:
    def __init__(self):
        self.pending_confirmation: Optional[str] = None
        self.last_command_time = 0
        self.command_cooldown = 1.0
        
    def process_multiple_commands(self, text: str) -> list:
        """Split and process multiple commands in one voice input"""
        # Split by 'and', 'then', commas
        commands = re.split(r'\s+(?:and|then)\s+|\s*,\s*', text.lower())
        results = []
        
        for cmd in commands:
            cmd = cmd.strip()
            if cmd:
                result = self.process_text(cmd)
                results.append(result)
                time.sleep(0.2)  # Small delay between commands
        
        return results
    
    def process_text(self, text: str) -> VoiceCommandResult:
        raw = text.strip()
        lowered = raw.lower()
        
        # Cooldown check
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return VoiceCommandResult(raw, "COOLDOWN", "WAIT", "Please wait...")
        self.last_command_time = current_time
        
        # File Management Commands
        if "create file" in lowered or "new file" in lowered:
            pyautogui.hotkey('ctrl', 'n')
            return VoiceCommandResult(raw, "FILE", "CREATE", "Created new file")
        
        elif "open file" in lowered:
            pyautogui.hotkey('ctrl', 'o')
            return VoiceCommandResult(raw, "FILE", "OPEN", "Opening file...")
        
        elif "close file" in lowered or "close document" in lowered:
            pyautogui.hotkey('ctrl', 'w')
            return VoiceCommandResult(raw, "FILE", "CLOSE", "Closed file")
        
        elif "minimize" in lowered:
            pyautogui.hotkey('alt', 'space')
            pyautogui.press('n')
            return VoiceCommandResult(raw, "WINDOW", "MINIMIZE", "Minimized window")
        
        elif "maximize" in lowered:
            pyautogui.hotkey('alt', 'space')
            pyautogui.press('x')
            return VoiceCommandResult(raw, "WINDOW", "MAXIMIZE", "Maximized window")
        
        # Document Formatting Commands
        elif "bold" in lowered:
            pyautogui.hotkey('ctrl', 'b')
            return VoiceCommandResult(raw, "FORMAT", "BOLD", "Applied bold")
        
        elif "italic" in lowered:
            pyautogui.hotkey('ctrl', 'i')
            return VoiceCommandResult(raw, "FORMAT", "ITALIC", "Applied italic")
        
        elif "underline" in lowered:
            pyautogui.hotkey('ctrl', 'u')
            return VoiceCommandResult(raw, "FORMAT", "UNDERLINE", "Applied underline")
        
        elif "align left" in lowered:
            pyautogui.hotkey('ctrl', 'l')
            return VoiceCommandResult(raw, "FORMAT", "ALIGN_LEFT", "Left aligned")
        
        elif "align center" in lowered:
            pyautogui.hotkey('ctrl', 'e')
            return VoiceCommandResult(raw, "FORMAT", "ALIGN_CENTER", "Center aligned")
        
        elif "align right" in lowered:
            pyautogui.hotkey('ctrl', 'r')
            return VoiceCommandResult(raw, "FORMAT", "ALIGN_RIGHT", "Right aligned")
        
        elif "font size" in lowered:
            # Extract number from command
            numbers = re.findall(r'\d+', lowered)
            if numbers:
                size = numbers[0]
                # Select all and change font size
                pyautogui.hotkey('ctrl', 'a')
                time.sleep(0.1)
                pyautogui.hotkey('ctrl', 'shift', 'p')
                time.sleep(0.1)
                pyautogui.typewrite(size)
                pyautogui.press('enter')
                return VoiceCommandResult(raw, "FORMAT", "FONT_SIZE", f"Font size set to {size}")
        
        # Copy, Cut, Paste Commands
        elif "copy" in lowered:
            pyautogui.hotkey('ctrl', 'c')
            return VoiceCommandResult(raw, "EDIT", "COPY", "Copied to clipboard")
        
        elif "cut" in lowered:
            pyautogui.hotkey('ctrl', 'x')
            return VoiceCommandResult(raw, "EDIT", "CUT", "Cut to clipboard")
        
        elif "paste" in lowered:
            pyautogui.hotkey('ctrl', 'v')
            return VoiceCommandResult(raw, "EDIT", "PASTE", "Pasted from clipboard")
        
        # Voice Typing
        elif "voice typing" in lowered or "start typing" in lowered:
            pyautogui.hotkey('win', 'h')
            return VoiceCommandResult(raw, "VOICE", "TYPING", "Voice typing activated")
        
        # Folder Management
        elif "create folder" in lowered or "new folder" in lowered:
            pyautogui.hotkey('ctrl', 'shift', 'n')
            return VoiceCommandResult(raw, "FOLDER", "CREATE", "Created new folder")
        
        elif "delete" in lowered and "folder" in lowered:
            pyautogui.press('delete')
            return VoiceCommandResult(raw, "FOLDER", "DELETE", "Deleted item")
        
        elif "rename" in lowered:
            pyautogui.press('f2')
            return VoiceCommandResult(raw, "FOLDER", "RENAME", "Ready to rename")
        
        elif "back" in lowered or "go back" in lowered:
            pyautogui.hotkey('alt', 'left')
            return VoiceCommandResult(raw, "NAV", "BACK", "Navigated back")
        
        # System Functions
        elif "shutdown" in lowered:
            self.pending_confirmation = "shutdown"
            return VoiceCommandResult(raw, "SYSTEM", "SHUTDOWN_PENDING", "Say confirm to shutdown")
        
        elif "sleep" in lowered:
            self.pending_confirmation = "sleep"
            return VoiceCommandResult(raw, "SYSTEM", "SLEEP_PENDING", "Say confirm to sleep")
        
        elif "confirm" in lowered or "yes" in lowered:
            return self._confirm(raw)
        
        elif "cancel" in lowered or "no" in lowered:
            return self._cancel(raw)
        
        elif "switch tab" in lowered:
            pyautogui.hotkey('ctrl', 'tab')
            return VoiceCommandResult(raw, "SYSTEM", "SWITCH_TAB", "Switched tab")
        
        elif "minimize all" in lowered:
            pyautogui.hotkey('win', 'd')
            return VoiceCommandResult(raw, "SYSTEM", "MINIMIZE_ALL", "Minimized all windows")
        
        elif "close all" in lowered:
            pyautogui.hotkey('alt', 'f4')
            return VoiceCommandResult(raw, "SYSTEM", "CLOSE_ALL", "Closing windows")
        
        # Scroll Commands
        elif "scroll up" in lowered:
            pyautogui.scroll(600)
            return VoiceCommandResult(raw, "SCROLL", "UP", "Scrolled up")
        
        elif "scroll down" in lowered:
            pyautogui.scroll(-600)
            return VoiceCommandResult(raw, "SCROLL", "DOWN", "Scrolled down")
        
        # Open Applications
        elif lowered.startswith("open "):
            target = lowered.replace("open ", "", 1).strip()
            return self._open_app(raw, target)
        
        return VoiceCommandResult(raw, "UNKNOWN", "NONE", "Command not recognized")
    
    def _open_app(self, raw: str, target: str) -> VoiceCommandResult:
        if not target:
            return VoiceCommandResult(raw, "OPEN_APP", "NONE", "Open what?")
        
        # Known folders
        folder_path = self._known_folder_path(target)
        if folder_path:
            os.startfile(folder_path)
            return VoiceCommandResult(raw, "OPEN_FOLDER", "OPEN_FOLDER", f"Opened folder: {target}")
        
        # Open applications
        pyautogui.press('win')
        time.sleep(0.2)
        pyautogui.typewrite(target)
        time.sleep(0.2)
        pyautogui.press('enter')
        return VoiceCommandResult(raw, "OPEN_APP", "OPEN_APP", f"Opening: {target}")
    
    @staticmethod
    def _known_folder_path(target: str) -> Optional[str]:
        key = re.sub(r'[^a-z]+', '', target.lower())
        mapping = {
            'downloads': str(Path.home() / 'Downloads'),
            'documents': str(Path.home() / 'Documents'),
            'pictures': str(Path.home() / 'Pictures'),
            'videos': str(Path.home() / 'Videos'),
            'desktop': str(Path.home() / 'Desktop'),
            'music': str(Path.home() / 'Music'),
        }
        return mapping.get(key)
    
    def _confirm(self, raw: str) -> VoiceCommandResult:
        if self.pending_confirmation == "shutdown":
            subprocess.Popen(['shutdown', '/s', '/t', '5'], shell=False)
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CONFIRM", "SHUTDOWN", "Shutting down in 5 seconds")
        elif self.pending_confirmation == "sleep":
            subprocess.Popen(['rundll32.exe', 'powrprof.dll,SetSuspendState', '0,1,0'], shell=False)
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CONFIRM", "SLEEP", "Sleeping...")
        return VoiceCommandResult(raw, "CONFIRM", "NONE", "Nothing to confirm")
    
    def _cancel(self, raw: str) -> VoiceCommandResult:
        if self.pending_confirmation == "shutdown":
            subprocess.Popen(['shutdown', '/a'], shell=False)
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CANCEL", "CANCEL_SHUTDOWN", "Shutdown cancelled")
        elif self.pending_confirmation == "sleep":
            self.pending_confirmation = None
            return VoiceCommandResult(raw, "CANCEL", "CANCEL_SLEEP", "Sleep cancelled")
        return VoiceCommandResult(raw, "CANCEL", "NONE", "Nothing to cancel")

# ==================== Vosk Voice Recognition ====================

class VoskStreamRecognizer:
    def __init__(self, model_path: Path, sample_rate: int = 16000):
        self.model = Model(str(model_path))
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.recognizer.SetWords(True)
        self.sample_rate = sample_rate
        self._queue = queue.Queue()
        self._run_flag = False
        self._thread = None
        self.is_active = False
        
    def _resolve_device(self) -> int:
        default_device = sd.default.device[0] if sd.default.device else None
        if isinstance(default_device, int) and default_device >= 0:
            info = sd.query_devices(default_device)
            if info.get('max_input_channels', 0) > 0:
                return default_device
        
        for idx, info in enumerate(sd.query_devices()):
            if info.get('max_input_channels', 0) > 0:
                return idx
        raise RuntimeError("No microphone found")
    
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            return
        self._queue.put(indata.copy())
    
    def start(self, on_text: Callable[[str], None], on_error: Optional[Callable[[str], None]] = None):
        if self._run_flag:
            return
        
        self._run_flag = True
        self.is_active = True
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(on_text, on_error),
            daemon=True
        )
        self._thread.start()
    
    def _run_loop(self, on_text: Callable[[str], None], on_error: Optional[Callable[[str], None]]):
        try:
            device = self._resolve_device()
            with sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.sample_rate,
                dtype='int16',
                blocksize=8000,
                callback=self._audio_callback
            ):
                while self._run_flag:
                    try:
                        data = self._queue.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    
                    if self.recognizer.AcceptWaveform(data.tobytes()):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()
                        if text:
                            on_text(text)
        except Exception as e:
            if on_error:
                on_error(f"Mic error: {e}")
        finally:
            self._run_flag = False
            self.is_active = False
    
    def stop(self):
        self._run_flag = False
        self.is_active = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

# ==================== Advanced Gesture Recognition ====================

class GestureRecognizer:
    def __init__(self, model_path: Path):
        self.model_data = joblib.load(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.gesture_labels = self.model_data['gesture_labels']
        self.reverse_labels = {v: k for k, v in self.gesture_labels.items()}
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Gesture action mapping for display only
        self.gesture_descriptions = {
            'L_closed_palm': 'Left Hand: Multiple Selection (CTRL)',
            'R_index': 'Right Hand: Cursor Mode',
            'R_open_palm': 'Right Hand: Open App (5 sec)',
            'R_closed_palm': 'Right Hand: Close App (5 sec)',
            'R_index_middle_up': 'Right Hand: Scroll Up',
            'R_index_middle_down': 'Right Hand: Scroll Down',
            'R_index_thumb_pinch': 'Right Hand: Left Click (1.5 sec)',
            'R_index_middle_pinch': 'Right Hand: Right Click (1.5 sec)',
            'L_index': 'Left Hand: Copy (1.5 sec)',
            'L_index_middle': 'Left Hand: Cut (1.5 sec)',
            'L_index_thumb_pinch': 'Left Hand: Paste (1.5 sec)',
            'L_R_palm_closed': 'Both Hands: Shutdown (5 sec)',
            'R_index_pinky': 'Right Hand: Switch Tab (2 sec)',
            'No_Hand': 'No Hand Detected'
        }
    
    def extract_features(self, image):
        """Extract features for prediction"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        features = []
        handedness = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = results.multi_handedness[idx].classification[0].label
                
                # Landmark coordinates
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
                
                # Distances from wrist
                wrist = hand_landmarks.landmark[0]
                for tip_idx in [4, 8, 12, 16, 20]:
                    tip = hand_landmarks.landmark[tip_idx]
                    distance = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
                    features.append(distance)
                
                # Finger angles
                for i in range(1, 5):
                    base = hand_landmarks.landmark[i * 4 + 1]
                    mid = hand_landmarks.landmark[i * 4 + 2]
                    tip = hand_landmarks.landmark[i * 4 + 3]
                    
                    v1 = np.array([mid.x - base.x, mid.y - base.y, mid.z - base.z])
                    v2 = np.array([tip.x - mid.x, tip.y - mid.y, tip.z - mid.z])
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                        features.append(angle)
                    else:
                        features.append(0)
                
                handedness.append(handedness_label)
        
        # Pad to fixed size
        max_features = 2 * (21 * 3 + 5 + 4)
        if len(features) < max_features:
            features.extend([0] * (max_features - len(features)))
        elif len(features) > max_features:
            features = features[:max_features]
        
        # Add hand count and handedness
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        features.append(hand_count)
        
        handedness_encoded = [0, 0]
        if 'Left' in handedness:
            handedness_encoded[0] = 1
        if 'Right' in handedness:
            handedness_encoded[1] = 1
        features.extend(handedness_encoded)
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, image) -> tuple:
        """Predict gesture without executing actions"""
        features = self.extract_features(image)
        
        if np.all(features == 0):  # No hand detected
            return 'No_Hand', 1.0, 'No Hand Detected'
        
        features_scaled = self.scaler.transform(features)
        
        # Get prediction with probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        gesture_name = self.reverse_labels.get(prediction, 'Unknown')
        description = self.gesture_descriptions.get(gesture_name, gesture_name)
        
        return gesture_name, confidence, description

# ==================== Modern GUI ====================

class ModernButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        
    def enterEvent(self, event):
        self.setGraphicsEffect(QGraphicsDropShadowEffect(
            blurRadius=20, color=QColor(0, 120, 212, 100), offset=QPointF(0, 0)
        ))
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.setGraphicsEffect(None)
        super().leaveEvent(event)

class ModernCircularProgress(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0
        self.setFixedSize(60, 60)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        
    def start_animation(self):
        self.value = 0
        self.timer.start(50)
        
    def stop_animation(self):
        self.timer.stop()
        self.value = 0
        self.update()
        
    def update_progress(self):
        self.value = (self.value + 5) % 360
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.rect().adjusted(5, 5, -5, -5)
        
        # Background circle
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(40, 40, 40))
        painter.drawEllipse(rect)
        
        # Progress arc
        painter.setPen(QPen(QColor(0, 120, 212), 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 0, self.value * 16)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = get_config()
        self.mode = 'voice'  # 'voice' or 'gesture'
        self.voice_active = False
        self.gesture_active = False
        self.camera_running = False
        self.screen_saver_mode = False
        
        # Initialize components
        self.voice_processor = VoiceCommandProcessor()
        self.voice_recognizer = None
        self.gesture_recognizer = None
        self.cap = None
        
        self.setup_ui()
        self.check_screen_status()
        
        # Screen saver detection timer
        self.screen_timer = QTimer()
        self.screen_timer.timeout.connect(self.check_screen_status)
        self.screen_timer.start(5000)  # Check every 5 seconds
        
    def setup_ui(self):
        self.setWindowTitle("Smart Gesture & Voice Control")
        self.setFixedSize(900, 700)
        
        # Set modern style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1e1e2e, stop:1 #2d2d44);
            }
            QLabel {
                color: #ffffff;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0078d4, stop:1 #106ebe);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #106ebe, stop:1 #005a9e);
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #107c10, stop:1 #0b5a0b);
            }
            QTextEdit, QListWidget {
                background-color: rgba(30, 30, 46, 0.9);
                color: #ffffff;
                border: 2px solid #3d3d5c;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }
            QGroupBox {
                color: #ffffff;
                font-size: 15px;
                font-weight: bold;
                border: 2px solid #3d3d5c;
                border-radius: 10px;
                margin-top: 15px;
                padding-top: 15px;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 10px 0 10px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QHBoxLayout()
        
        title = QLabel("🤖 Smart Control System")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #ffffff;
            padding: 10px;
            background: rgba(0, 120, 212, 0.3);
            border-radius: 15px;
        """)
        header.addWidget(title)
        
        # Status indicator
        self.status_indicator = ModernCircularProgress()
        header.addWidget(self.status_indicator, alignment=Qt.AlignRight)
        
        layout.addLayout(header)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.setSpacing(20)
        
        self.voice_btn = ModernButton("🎤 Voice Control")
        self.voice_btn.setCheckable(True)
        self.voice_btn.setChecked(True)
        self.voice_btn.clicked.connect(lambda: self.set_mode('voice'))
        
        self.gesture_btn = ModernButton("✋ Gesture Control")
        self.gesture_btn.setCheckable(True)
        self.gesture_btn.clicked.connect(lambda: self.set_mode('gesture'))
        
        mode_layout.addWidget(self.voice_btn)
        mode_layout.addWidget(self.gesture_btn)
        mode_layout.addStretch()
        
        layout.addLayout(mode_layout)
        
        # Camera preview (hidden by default in voice mode)
        self.camera_group = QGroupBox("Camera Preview")
        self.camera_layout = QVBoxLayout()
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(320, 240)
        self.camera_label.setStyleSheet("""
            background-color: #2d2d44;
            border: 3px solid #3d3d5c;
            border-radius: 10px;
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("📷 Camera Off")
        self.camera_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        self.camera_group.setLayout(self.camera_layout)
        self.camera_group.hide()
        
        layout.addWidget(self.camera_group)
        
        # Command display
        command_group = QGroupBox("Command")
        command_layout = QVBoxLayout()
        
        self.command_display = QTextEdit()
        self.command_display.setMaximumHeight(80)
        self.command_display.setPlaceholderText("Waiting for command...")
        self.command_display.setReadOnly(True)
        command_layout.addWidget(self.command_display)
        
        command_group.setLayout(command_layout)
        layout.addWidget(command_group)
        
        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_display = QTextEdit()
        self.status_display.setMaximumHeight(80)
        self.status_display.setPlaceholderText("System ready...")
        self.status_display.setReadOnly(True)
        status_layout.addWidget(self.status_display)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Gesture/Command history
        history_group = QGroupBox("Activity History")
        history_layout = QVBoxLayout()
        
        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(150)
        history_layout.addWidget(self.history_list)
        
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)
        
        # System info
        info_layout = QHBoxLayout()
        
        self.mic_status = QLabel("🎤 Microphone: Off")
        self.mic_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        
        self.camera_status = QLabel("📷 Camera: Off")
        self.camera_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        
        info_layout.addWidget(self.mic_status)
        info_layout.addWidget(self.camera_status)
        info_layout.addStretch()
        
        layout.addLayout(info_layout)
        
        # Initialize voice
        self.init_voice()
        
        # Show window
        self.show()
        self.add_to_history("System", "System initialized", "Ready")
        
    def init_voice(self):
        """Initialize voice recognition"""
        try:
            self.voice_recognizer = VoskStreamRecognizer(
                model_path=self.cfg.vosk_model_path
            )
            self.add_to_history("System", "Voice recognition initialized", "Success")
        except Exception as e:
            self.add_to_history("System", f"Voice init failed: {e}", "Error")
    
    def init_gesture(self):
        """Initialize gesture recognition"""
        try:
            if self.cfg.gesture_model_path.exists():
                self.gesture_recognizer = GestureRecognizer(self.cfg.gesture_model_path)
                self.add_to_history("System", "Gesture recognition initialized", "Success")
            else:
                self.add_to_history("System", 
                    "Gesture model not found. Please train the model first using train_gesture_model.py", 
                    "Warning")
        except Exception as e:
            self.add_to_history("System", f"Gesture init failed: {e}", "Error")
    
    def set_mode(self, mode):
        """Switch between voice and gesture mode"""
        self.mode = mode
        
        if mode == 'voice':
            self.voice_btn.setChecked(True)
            self.gesture_btn.setChecked(False)
            self.camera_group.hide()
            self.stop_gesture()
            self.start_voice()
            self.add_to_history("Mode", "Switched to Voice Control", "Info")
        else:
            self.voice_btn.setChecked(False)
            self.gesture_btn.setChecked(True)
            self.camera_group.show()
            self.stop_voice()
            self.init_gesture()
            self.start_gesture()
            self.add_to_history("Mode", "Switched to Gesture Control", "Info")
    
    def start_voice(self):
        """Start voice recognition"""
        if self.voice_active or self.screen_saver_mode:
            return
        
        if not self.voice_recognizer:
            self.init_voice()
        
        try:
            self.voice_recognizer.start(
                on_text=self.on_voice_command,
                on_error=self.on_voice_error
            )
            self.voice_active = True
            self.mic_status.setText("🎤 Microphone: Active")
            self.mic_status.setStyleSheet("color: #44ff44; font-weight: bold;")
            self.status_indicator.start_animation()
            self.add_to_history("Voice", "Listening...", "Active")
        except Exception as e:
            self.add_to_history("Voice", f"Failed to start: {e}", "Error")
    
    def stop_voice(self):
        """Stop voice recognition"""
        if self.voice_recognizer:
            self.voice_recognizer.stop()
        self.voice_active = False
        self.mic_status.setText("🎤 Microphone: Off")
        self.mic_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.status_indicator.stop_animation()
    
    def start_gesture(self):
        """Start gesture recognition"""
        if self.gesture_active or self.screen_saver_mode or not self.gesture_recognizer:
            return
        
        self.gesture_active = True
        self.camera_running = True
        self.camera_status.setText("📷 Camera: Active")
        self.camera_status.setStyleSheet("color: #44ff44; font-weight: bold;")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def stop_gesture(self):
        """Stop gesture recognition"""
        self.gesture_active = False
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.camera_status.setText("📷 Camera: Off")
        self.camera_status.setStyleSheet("color: #ff4444; font-weight: bold;")
        self.camera_label.setPixmap(QPixmap())
        self.camera_label.setText("📷 Camera Off")
    
    def camera_loop(self):
        """Camera capture and gesture recognition loop"""
        self.cap = cv2.VideoCapture(self.cfg.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.camera_running and self.gesture_active and not self.screen_saver_mode:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Predict gesture
            if self.gesture_recognizer:
                gesture, confidence, description = self.gesture_recognizer.predict(frame)
                
                # Display on frame
                cv2.putText(frame, f"{gesture}: {confidence:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, description, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Update UI with gesture info (no actions executed)
                if confidence > 0.7:
                    self.update_gesture_display.emit(gesture, description, confidence)
            
            # Convert to QPixmap for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Update camera label (thread-safe)
            self.update_camera_display.emit(pixmap)
            
            time.sleep(0.03)  # ~30 FPS
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    # Custom signals for thread-safe UI updates
    update_camera_display = pyqtSignal(QPixmap)
    update_gesture_display = pyqtSignal(str, str, float)
    update_voice_display = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        # ... (previous init code)
        
        # Connect signals
        self.update_camera_display.connect(self.on_camera_frame)
        self.update_gesture_display.connect(self.on_gesture_detected)
        self.update_voice_display.connect(self.on_voice_detected)
    
    def on_camera_frame(self, pixmap):
        """Update camera display"""
        self.camera_label.setPixmap(pixmap)
        self.camera_label.setText("")
    
    def on_gesture_detected(self, gesture, description, confidence):
        """Handle gesture detection (display only, no actions)"""
        self.command_display.setText(f"🎯 Gesture: {gesture}\n📝 {description}")
        self.status_display.setText(f"Confidence: {confidence:.2%}")
        
        # Add to history
        self.add_to_history("Gesture", f"{gesture} ({confidence:.1%})", description)
    
    def on_voice_command(self, text):
        """Handle voice command"""
        # Process multiple commands
        results = self.voice_processor.process_multiple_commands(text)
        
        for result in results:
            self.command_display.setText(f"🎤 Command: {result.text}")
            self.status_display.setText(f"Status: {result.status}")
            
            # Add to history
            self.add_to_history("Voice", result.text, result.status)
            
            # Play sound for confirmation
            QApplication.beep()
    
    def on_voice_error(self, error):
        """Handle voice recognition error"""
        self.add_to_history("Voice", f"Error: {error}", "Error")
    
    def on_voice_detected(self, command, status):
        """Update voice display"""
        self.command_display.setText(f"🎤 Command: {command}")
        self.status_display.setText(f"Status: {status}")
    
    def check_screen_status(self):
        """Check if screen is off or system is in sleep mode"""
        import ctypes
        try:
            # Check if screen saver is active or display is off
            is_screensaver_active = ctypes.windll.user32.SystemParametersInfoW(0x0073, 0, 0, 0)
            
            if is_screensaver_active:
                if not self.screen_saver_mode:
                    self.screen_saver_mode = True
                    self.stop_voice()
                    self.stop_gesture()
                    self.add_to_history("System", "Screen off/sleep detected - Paused all input", "Info")
            else:
                if self.screen_saver_mode:
                    self.screen_saver_mode = False
                    self.add_to_history("System", "Screen on - Resuming", "Info")
                    # Restore previous mode
                    if self.mode == 'voice':
                        self.start_voice()
                    else:
                        self.start_gesture()
        except:
            pass
    
    def add_to_history(self, source, command, status):
        """Add entry to history list"""
        timestamp = QTime.currentTime().toString("hh:mm:ss")
        
        # Set color based on status
        color = "#ffffff"  # default white
        if "Error" in status:
            color = "#ff4444"
        elif "Success" in status or "Active" in status:
            color = "#44ff44"
        elif "Warning" in status:
            color = "#ffff44"
        
        item = QListWidgetItem(f"[{timestamp}] {source}: {command} -> {status}")
        item.setForeground(QColor(color))
        
        self.history_list.insertItem(0, item)
        if self.history_list.count() > 20:
            self.history_list.takeItem(20)
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.stop_voice()
        self.stop_gesture()
        self.screen_timer.stop()
        event.accept()

# ==================== Main ====================

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application icon
    app.setWindowIcon(QIcon())
    
    window = MainWindow()
    
    # Check for required models
    cfg = get_config()
    if not cfg.vosk_model_path.exists():
        QMessageBox.warning(window, "Model Missing",
            "Vosk model not found. Please download it from:\n"
            "https://alphacephei.com/vosk/models\n\n"
            f"Expected path: {cfg.vosk_model_path}")
    
    if not cfg.gesture_model_path.exists():
        QMessageBox.warning(window, "Model Missing",
            "Gesture model not found. Please train it first:\n"
            "python train_gesture_model.py")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
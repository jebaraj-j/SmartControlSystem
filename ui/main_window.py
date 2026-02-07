from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import sys
import time

from gesture.detector import HandDetector
from gesture.virtual_mouse import VirtualMouse


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    status_update_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.detector = HandDetector(max_hands=2)
        self.vm = VirtualMouse()
        self.start_time = time.perf_counter()
        self.current_mode = "Gesture"
        self.current_context = "System"
        self.hands_detected = 0
        self.last_action = "None"
        self.current_gesture = "None"
        self.previous_gesture = "None"
        self.gesture_hold_frames = 0
        self.gesture_hold_threshold = 5  # Reduced from 8 to 5 for faster response

        # Track scroll gesture position for direction detection
        self.prev_peace_y = None
        self.scroll_gesture_active = False

        # Action performed flag to prevent repeats
        self.action_performed = False

    def run(self):
        cap = cv2.VideoCapture(0)
        frame_count = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            timestamp_ms = int((time.perf_counter() - self.start_time) * 1000)
            frame, results = self.detector.find_hands(frame, timestamp_ms)
            self.hands_detected = len(results.hand_landmarks) if results.hand_landmarks else 0

            if results.hand_landmarks:
                # Find right hand (primary for cursor)
                right_hand_landmarks = None
                for idx, hand_lms in enumerate(results.hand_landmarks):
                    if results.handedness and idx < len(results.handedness):
                        if results.handedness[idx][0].category_name == "Right":
                            right_hand_landmarks = hand_lms
                            break

                if right_hand_landmarks and self.current_mode == "Gesture":
                    index_tip = right_hand_landmarks[8]
                    h, w, _ = frame.shape

                    # Always move cursor
                    self.vm.move_cursor(index_tip, w, h)

                    # Detect gesture
                    gesture = self.detect_gesture(right_hand_landmarks)
                    self.current_gesture = gesture

                    # Display gesture info on frame for debugging
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Hold: {self.gesture_hold_frames}/{self.gesture_hold_threshold}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Gesture hold logic
                    if gesture == self.previous_gesture and gesture != "None":
                        self.gesture_hold_frames += 1

                        # Trigger action after holding threshold
                        if self.gesture_hold_frames >= self.gesture_hold_threshold and not self.action_performed:
                            success = self.perform_gesture_action(gesture)
                            if success:
                                self.last_action = f"✓ {gesture}"
                                self.action_performed = True  # Mark as performed
                                cv2.putText(frame, f"ACTION: {gesture}!", (10, 90),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            else:
                                self.last_action = f"⏳ {gesture} (cooldown)"
                        else:
                            self.last_action = f"Holding: {gesture} ({self.gesture_hold_frames}/{self.gesture_hold_threshold})"
                    else:
                        # Gesture changed, reset counter
                        self.gesture_hold_frames = 0
                        self.action_performed = False
                        if gesture == "None":
                            self.last_action = "Cursor Move"
                        else:
                            self.last_action = f"Detecting: {gesture}"

                    self.previous_gesture = gesture
                else:
                    self.last_action = "Waiting for Right Hand"
                    self.current_gesture = "None"
                    self.gesture_hold_frames = 0
                    self.prev_peace_y = None
                    self.action_performed = False
            else:
                self.last_action = "No Hands Detected"
                self.current_gesture = "None"
                self.gesture_hold_frames = 0
                self.prev_peace_y = None
                self.action_performed = False

            # Convert frame to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(convert_to_qt.copy())

            # Emit status
            status = {
                "mode": self.current_mode,
                "context": self.current_context,
                "hands": self.hands_detected,
                "last_action": self.last_action,
                "gesture": self.current_gesture,
                "hold_frames": self.gesture_hold_frames
            }
            self.status_update_signal.emit(status)

        cap.release()

    def detect_gesture(self, hand_landmarks):
        """
        Gesture detection with improved thresholds
        """
        try:
            # Key landmarks
            thumb_tip = hand_landmarks[4]
            thumb_ip = hand_landmarks[3]
            index_tip = hand_landmarks[8]
            index_pip = hand_landmarks[6]
            middle_tip = hand_landmarks[12]
            middle_pip = hand_landmarks[10]
            ring_tip = hand_landmarks[16]
            ring_pip = hand_landmarks[14]
            pinky_tip = hand_landmarks[20]
            pinky_pip = hand_landmarks[18]

            # Calculate distances
            def distance(p1, p2):
                return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

            thumb_index_dist = distance(thumb_tip, index_tip)
            thumb_middle_dist = distance(thumb_tip, middle_tip)

            # Check if fingers are extended (tip above PIP joint in Y coordinate)
            index_extended = index_tip.y < index_pip.y
            middle_extended = middle_tip.y < middle_pip.y
            ring_extended = ring_tip.y < ring_pip.y
            pinky_extended = pinky_tip.y < pinky_pip.y

            # Debug: print distances occasionally
            if int(time.time() * 10) % 10 == 0:  # Every 0.1 seconds
                print(f"Thumb-Index: {thumb_index_dist:.3f}, Thumb-Middle: {thumb_middle_dist:.3f}")

            # GESTURE 1: Left Click - Index + Thumb Pinch
            # More lenient threshold for easier detection
            if thumb_index_dist < 0.08:  # Increased from 0.05 to 0.08
                return "Left_Click"

            # GESTURE 2: Right Click - Middle + Thumb Pinch
            # Middle and thumb close, index far away
            elif thumb_middle_dist < 0.08 and thumb_index_dist > 0.1:
                return "Right_Click"

            # GESTURE 3 & 4: Scroll - Index + Middle extended (Peace sign)
            elif index_extended and middle_extended and not ring_extended and not pinky_extended:
                # Calculate average Y position of index and middle fingers
                current_peace_y = (index_tip.y + middle_tip.y) / 2

                if self.prev_peace_y is not None:
                    # Calculate movement (positive = down, negative = up)
                    y_movement = current_peace_y - self.prev_peace_y

                    # More sensitive threshold for scroll detection
                    if abs(y_movement) > 0.008:  # Reduced from 0.01 to 0.008
                        if y_movement < 0:  # Moving up
                            self.prev_peace_y = current_peace_y
                            return "Scroll_Up"
                        else:  # Moving down
                            self.prev_peace_y = current_peace_y
                            return "Scroll_Down"

                # Store current position for next frame
                self.prev_peace_y = current_peace_y
                return "Peace_Hold"  # Peace sign held steady

            else:
                self.prev_peace_y = None  # Reset if not in peace sign
                return "None"

        except Exception as e:
            print(f"Gesture detection error: {e}")
            self.prev_peace_y = None
            return "None"

    def perform_gesture_action(self, gesture_name):
        """
        Execute action based on detected gesture with cooldown
        Returns True if action was performed, False if on cooldown
        """
        try:
            if gesture_name == "Left_Click":
                result = self.vm.click()
                if result:
                    print("✓ LEFT CLICK PERFORMED!")
                return result

            elif gesture_name == "Right_Click":
                result = self.vm.right_click()
                if result:
                    print("✓ RIGHT CLICK PERFORMED!")
                return result

            elif gesture_name == "Scroll_Up":
                result = self.vm.scroll_up(5)  # Increased scroll amount
                if result:
                    print("✓ SCROLL UP")
                return result

            elif gesture_name == "Scroll_Down":
                result = self.vm.scroll_down(5)  # Increased scroll amount
                if result:
                    print("✓ SCROLL DOWN")
                return result

            return False

        except Exception as e:
            print(f"Error performing gesture action {gesture_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        self._run_flag = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Control System - Gesture Control")
        self.setGeometry(100, 100, 1400, 800)

        # Apply modern dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left: Video feed container
        video_container = QFrame()
        video_container.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                border: 2px solid #4CAF50;
            }
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(10, 10, 10, 10)

        # Video title
        video_title = QLabel("🎥 Live Camera Feed (With Gesture Debug)")
        video_title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #4CAF50;
            padding: 8px;
        """)
        video_title.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(video_title)

        # Video feed
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #000000;
            border-radius: 5px;
            min-height: 480px;
            min-width: 640px;
        """)
        video_layout.addWidget(self.video_label)

        # Right side - Two panels stacked
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)

        # Status panel
        status_frame = self.create_status_panel()
        right_layout.addWidget(status_frame, 3)

        # Gesture guide panel
        guide_frame = self.create_gesture_guide()
        right_layout.addWidget(guide_frame, 2)

        # Add to main layout
        main_layout.addWidget(video_container, 3)
        main_layout.addLayout(right_layout, 2)

        # Thread
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_update_signal.connect(self.update_status)

        self.system_running = False

    def create_status_panel(self):
        """Create the status information panel"""
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                border: 2px solid #3d3d3d;
            }
        """)
        status_layout = QVBoxLayout(status_frame)
        status_layout.setContentsMargins(15, 15, 15, 15)
        status_layout.setSpacing(10)

        # Title
        title_label = QLabel("📊 System Status")
        title_label.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            padding: 10px;
            color: #4CAF50;
            background-color: #3d3d3d;
            border-radius: 5px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(title_label)

        # Status labels
        self.mode_label = QLabel("🎮 Mode: Gesture")
        self.context_label = QLabel("💻 Context: System")
        self.hands_label = QLabel("✋ Hands: 0")
        self.gesture_label = QLabel("👆 Gesture: None")
        self.action_label = QLabel("⚡ Action: None")
        self.hold_label = QLabel("⏱️ Hold: 0/5 frames")

        label_style = """
            font-size: 15px;
            padding: 10px;
            background-color: #3d3d3d;
            border-radius: 8px;
            border-left: 4px solid #4CAF50;
            color: #e0e0e0;
        """

        for label in [self.mode_label, self.context_label, self.hands_label,
                      self.gesture_label, self.action_label, self.hold_label]:
            label.setStyleSheet(label_style)
            status_layout.addWidget(label)

        status_layout.addStretch()

        # Control button
        self.start_button = QPushButton("▶ Start System")
        self.start_button.clicked.connect(self.toggle_system)
        self.start_button.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                padding: 15px;
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        status_layout.addWidget(self.start_button)

        return status_frame

    def create_gesture_guide(self):
        """Create gesture guide panel"""
        guide_frame = QFrame()
        guide_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
                border: 2px solid #3d3d3d;
            }
        """)
        guide_layout = QVBoxLayout(guide_frame)
        guide_layout.setContentsMargins(15, 15, 15, 15)
        guide_layout.setSpacing(8)

        # Title
        guide_title = QLabel("📖 Gesture Guide")
        guide_title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            padding: 8px;
            color: #2196F3;
            background-color: #3d3d3d;
            border-radius: 5px;
        """)
        guide_title.setAlignment(Qt.AlignCenter)
        guide_layout.addWidget(guide_title)

        # Gesture instructions
        gestures = [
            ("👌 Index + Thumb Pinch", "Left Click", "#4CAF50"),
            ("🤏 Middle + Thumb Pinch", "Right Click", "#FF9800"),
            ("✌️⬆ Peace Sign + Move UP", "Scroll Up", "#2196F3"),
            ("✌️⬇ Peace Sign + Move DOWN", "Scroll Down", "#9C27B0"),
        ]

        for gesture, action, color in gestures:
            gesture_label = QLabel(f"<b>{gesture}</b><br>→ {action}")
            gesture_label.setStyleSheet(f"""
                font-size: 13px;
                padding: 10px;
                background-color: #3d3d3d;
                border-radius: 5px;
                color: #b0b0b0;
                border-left: 3px solid {color};
            """)
            gesture_label.setWordWrap(True)
            guide_layout.addWidget(gesture_label)

        # Tips
        tip_label = QLabel("💡 <b>Tips:</b><br>"
                           "• Hold gesture steady for 0.2s<br>"
                           "• Watch console for action confirmations<br>"
                           "• Pinch fingers closer for better detection<br>"
                           "• Check camera feed for gesture debug info")
        tip_label.setStyleSheet("""
            font-size: 11px;
            padding: 10px;
            background-color: #3d3d3d;
            border-radius: 5px;
            color: #FFC107;
            margin-top: 5px;
            border-left: 3px solid #FFC107;
        """)
        tip_label.setWordWrap(True)
        guide_layout.addWidget(tip_label)

        guide_layout.addStretch()

        return guide_frame

    def toggle_system(self):
        if not self.system_running:
            self.thread.start()
            self.start_button.setText("⏹ Stop System")
            self.start_button.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 15px;
                    background-color: #f44336;
                    color: white;
                    border-radius: 10px;
                    font-weight: bold;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #da190b;
                }
                QPushButton:pressed {
                    background-color: #c41408;
                }
            """)
            self.system_running = True
            print("\n" + "=" * 50)
            print("SYSTEM STARTED - Watch console for action confirmations!")
            print("=" * 50 + "\n")
        else:
            self.thread.stop()
            self.start_button.setText("▶ Start System")
            self.start_button.setStyleSheet("""
                QPushButton {
                    font-size: 18px;
                    padding: 15px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    font-weight: bold;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #3d8b40;
                }
            """)
            self.system_running = False

    def update_image(self, qt_img):
        scaled_img = qt_img.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(QPixmap.fromImage(scaled_img))

    def update_status(self, status):
        self.mode_label.setText(f"🎮 Mode: {status['mode']}")
        self.context_label.setText(f"💻 Context: {status['context']}")
        self.hands_label.setText(f"✋ Hands: {status['hands']}")

        gesture = status.get('gesture', 'None')
        gesture_display = gesture
        if gesture == "Peace_Hold":
            gesture_display = "✌️ Peace (Move to Scroll)"

        self.gesture_label.setText(f"👆 Gesture: {gesture_display}")
        self.action_label.setText(f"⚡ Action: {status['last_action']}")

        hold_frames = status.get('hold_frames', 0)
        self.hold_label.setText(f"⏱️ Hold: {hold_frames}/5 frames")

    def closeEvent(self, event):
        """Ensure thread stops when window closes"""
        if self.system_running:
            self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
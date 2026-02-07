import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand connections (21 landmarks)
hand_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


class HandDetector:
    def __init__(self, max_hands=2, model_path='hand_landmarker.task'):
        base_options = python.BaseOptions
        hand_landmarker = vision.HandLandmarker
        hand_landmarker_options = vision.HandLandmarkerOptions
        vision_running_mode = mp.tasks.vision.RunningMode

        self.options = hand_landmarker_options(
            base_options=base_options(model_asset_path=model_path),
            running_mode=vision_running_mode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = hand_landmarker.create_from_options(self.options)

    def find_hands(self, frame, timestamp_ms):
        # CLAHE pre-processing for better lighting
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw hands
        if results.hand_landmarks:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):  # hand_landmarks = list of 21 landmarks
                # Draw landmarks (blue dots)
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 8, (255, 0, 0), -1)

                # Draw connections (thick green lines)
                for connection in hand_connections:
                    start = hand_landmarks[connection[0]]
                    end = hand_landmarks[connection[1]]
                    x1, y1 = int(start.x * frame.shape[1]), int(start.y * frame.shape[0])
                    x2, y2 = int(end.x * frame.shape[1]), int(end.y * frame.shape[0])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

                # Left/Right label (yellow, big)
                if results.handedness and idx < len(results.handedness):
                    handedness_label = results.handedness[idx][0].category_name
                else:
                    handedness_label = "Unknown"
                wrist = hand_landmarks[0]  # Wrist landmark
                cv2.putText(frame, handedness_label,
                            (int(wrist.x * frame.shape[1]) - 30, int(wrist.y * frame.shape[0]) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        return frame, results
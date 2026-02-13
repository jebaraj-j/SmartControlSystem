# train_gesture_model.py
import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import albumentations as A
from pathlib import Path
import joblib

class GestureTrainer:
    def __init__(self, dataset_path="gesture_dataset"):
        self.dataset_path = Path(dataset_path)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Define gestures based on document
        self.gesture_labels = {
            'L_closed_palm': 0,      # Left hand - multiple selection (CTRL)
            'R_index': 1,            # Right index - cursor mode
            'R_open_palm': 2,        # Right open palm - open app
            'R_closed_palm': 3,      # Right closed palm - close app
            'R_index_middle_up': 4,  # Right index and middle up - scroll up
            'R_index_middle_down': 5, # Right index and middle down - scroll down
            'R_index_thumb_pinch': 6, # Right index and thumb pinch - left click
            'R_index_middle_pinch': 7, # Right index and middle pinch - right click
            'L_index': 8,            # Left index only - copy
            'L_index_middle': 9,     # Left index and middle - cut
            'L_index_thumb_pinch': 10, # Left index and thumb pinch - paste
            'L_R_palm_closed': 11,   # Left and Right palm closed - shutdown
            'R_index_pinky': 12,     # Right index and pinky - switch tab
            'No_Hand': 13
        }
        
        # Advanced data augmentation pipeline
        self.augmentation = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomScale(scale_limit=(-0.3, 0.2), p=0.4),
            A.Rotate(limit=30, p=0.3),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.IAAPerspective(scale=(0.02, 0.05), p=0.2),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        ])
        
    def extract_hand_features(self, image):
        """Extract comprehensive hand features from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        features = []
        handedness = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_label = results.multi_handedness[idx].classification[0].label
                
                # Extract normalized landmark coordinates
                for landmark in hand_landmarks.landmark:
                    features.extend([landmark.x, landmark.y, landmark.z])
                
                # Calculate distances between key points
                wrist = hand_landmarks.landmark[0]
                for tip_idx in [4, 8, 12, 16, 20]:  # Thumb to pinky tips
                    tip = hand_landmarks.landmark[tip_idx]
                    distance = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2 + (tip.z - wrist.z)**2)
                    features.append(distance)
                
                # Calculate angles between fingers
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
        
        # Pad features to fixed size (assuming max 2 hands, 21 landmarks * 3 + 5 distances + 4 angles)
        max_features = 2 * (21 * 3 + 5 + 4)
        if len(features) < max_features:
            features.extend([0] * (max_features - len(features)))
        elif len(features) > max_features:
            features = features[:max_features]
        
        # Add hand count and handedness info
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        features.extend([hand_count])
        
        handedness_encoded = [0, 0]  # [is_left, is_right]
        if 'Left' in handedness:
            handedness_encoded[0] = 1
        if 'Right' in handedness:
            handedness_encoded[1] = 1
        features.extend(handedness_encoded)
        
        return np.array(features)
    
    def load_and_augment_data(self):
        """Load images and apply data augmentation"""
        X = []
        y = []
        
        print("Loading dataset and applying augmentation...")
        
        for gesture_name, gesture_label in self.gesture_labels.items():
            gesture_folder = self.dataset_path / gesture_name
            if not gesture_folder.exists():
                print(f"Warning: {gesture_folder} not found")
                continue
            
            image_files = list(gesture_folder.glob("*.jpg")) + list(gesture_folder.glob("*.png"))
            print(f"Found {len(image_files)} original images for {gesture_name}")
            
            for img_path in image_files:
                # Load original image
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Extract features from original
                features = self.extract_hand_features(image)
                if np.any(features):  # If hand detected
                    X.append(features)
                    y.append(gesture_label)
                
                # Generate augmented versions
                for i in range(5):  # 5 augmented versions per image
                    augmented = self.augmentation(image=image)['image']
                    aug_features = self.extract_hand_features(augmented)
                    if np.any(aug_features):
                        X.append(aug_features)
                        y.append(gesture_label)
        
        return np.array(X), np.array(y)
    
    def train(self):
        """Train the gesture recognition model"""
        # Load and augment data
        X, y = self.load_and_augment_data()
        
        if len(X) == 0:
            print("No training data found!")
            return
        
        print(f"Total samples after augmentation: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
        )
        
        print("Training model...")
        grid_search.fit(X_train_balanced, y_train_balanced)
        
        # Evaluate
        y_pred = grid_search.predict(X_test_scaled)
        
        print("\nBest Parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(self.gesture_labels.keys())))
        
        # Save model and scaler
        model_data = {
            'model': grid_search.best_estimator_,
            'scaler': scaler,
            'gesture_labels': self.gesture_labels,
            'feature_size': X.shape[1]
        }
        
        joblib.dump(model_data, 'gesture_model.pkl')
        print("\nModel saved as 'gesture_model.pkl'")

if __name__ == "__main__":
    trainer = GestureTrainer()
    trainer.train()
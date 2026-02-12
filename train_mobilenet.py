"""
train_mobilenet.py — Train MobileNetV2 on gesture images
"""

import os, sys, json, time
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import GESTURE_CLASSES, NUM_CLASSES


# ─── config ───────────────────────────────────────────────
DATASET_ROOT     = "dataset"
IMAGE_SIZE       = (224, 224)
BATCH_SIZE       = 32
EPOCHS_PHASE1    = 40
EPOCHS_PHASE2    = 15
LR_PHASE1        = 0.001
LR_PHASE2        = 0.0001
VAL_SPLIT        = 0.2
MODEL_OUT        = "gesture_model.h5"
CLASS_NAMES_OUT  = "class_names.json"


# ─── 1. load images ───────────────────────────────────────
def load_dataset():
    imgs, labels = [], []
    print("\n📂 Loading dataset…")

    for idx, name in enumerate(GESTURE_CLASSES):
        folder = os.path.join(DATASET_ROOT, name)
        if not os.path.isdir(folder):
            print(f"  ⚠ {name} missing")
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
        for f in files:
            im = cv2.imread(os.path.join(folder, f))
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
            imgs.append(im)
            labels.append(idx)

        print(f"  ✓ {name:20s} {len(files):4d} images")

    return np.array(imgs), np.array(labels)


# ─── 2. preprocess + split ────────────────────────────────
def preprocess(imgs, labels):
    imgs = imgs.astype("float32") / 255.0
    labels_oh = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    return imgs, labels_oh


def split(imgs, labels_oh):
    Xtr, Xv, ytr, yv = train_test_split(
        imgs, labels_oh, test_size=VAL_SPLIT,
        random_state=42, stratify=np.argmax(labels_oh, axis=1)
    )
    print(f"\nTrain {len(Xtr)}  |  Val {len(Xv)}")
    return Xtr, Xv, ytr, yv


# ─── 3. augmentation (FIXED) ──────────────────────────────
def generators(Xtr, ytr, Xv, yv):

    train_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
        # ❌ NO horizontal_flip
    )

    val_gen = ImageDataGenerator()

    return (
        train_gen.flow(Xtr, ytr, batch_size=BATCH_SIZE, shuffle=True),
        val_gen.flow(Xv, yv, batch_size=BATCH_SIZE, shuffle=False)
    )


# ─── 4. build model ───────────────────────────────────────
def build():
    print("\n🤖 Building MobileNetV2")

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(base.input, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base


# ─── callbacks ────────────────────────────────────────────
def callbacks():
    return [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT, save_best_only=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]


# ─── training ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("   MOBILENET GESTURE TRAINING")
    print("=" * 55)

    imgs, labels = load_dataset()
    imgs, labels_oh = preprocess(imgs, labels)
    Xtr, Xv, ytr, yv = split(imgs, labels_oh)
    tg, vg = generators(Xtr, ytr, Xv, yv)

    model, base = build()

    print("\n🚀 Phase 1")
    model.fit(tg, validation_data=vg,
              epochs=EPOCHS_PHASE1,
              callbacks=callbacks())

    print("\n🔧 Phase 2 (fine-tune)")
    for layer in base.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(tg, validation_data=vg,
              epochs=EPOCHS_PHASE2,
              callbacks=callbacks())

    print("\n📊 Evaluation")
    pred = np.argmax(model.predict(Xv), axis=1)
    truth = np.argmax(yv, axis=1)

    print(classification_report(truth, pred, target_names=GESTURE_CLASSES))
    print(confusion_matrix(truth, pred))

    model.save(MODEL_OUT)
    with open(CLASS_NAMES_OUT, "w") as f:
        json.dump(GESTURE_CLASSES, f, indent=2)

    print("\n✅ Training complete")


if __name__ == "__main__":
    main()

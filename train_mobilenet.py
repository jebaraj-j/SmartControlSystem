"""
train_mobilenet.py  —  Train MobileNetV2 on your collected gesture images.
===========================================================================
Phase 1 : freeze backbone, train Dense head only   (fast, ~5 min)
Phase 2 : unfreeze last 30 layers, fine-tune       (slower, higher accuracy)

Imports class names from labels.py so it stays in sync with collection
and inference.

Output files:
    gesture_model.h5     — saved Keras model
    class_names.json     — list of gesture labels (same order as CNN output)
    training_history.png — accuracy / loss curves
"""

import os, sys, json, time
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (GlobalAveragePooling2D,Dense, Dropout, BatchNormalization)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib
matplotlib.use('Agg')
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
            print(f"  ⚠  {name} — folder missing, skipped")
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith('.jpg'))
        if not files:
            print(f"  ⚠  {name} — empty, skipped")
            continue
        for f in files:
            im = cv2.imread(os.path.join(folder, f))
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
            imgs.append(im)
            labels.append(idx)
        print(f"  ✓  {name:20s}  {len(files):>4} images  (label {idx})")
    return np.array(imgs), np.array(labels)


# ─── 2. preprocess + split ────────────────────────────────
def preprocess(imgs, labels):
    imgs = imgs.astype('float32') / 255.0
    labels_oh = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)
    return imgs, labels_oh

def split(imgs, labels):
    Xtr, Xv, ytr, yv = train_test_split(
        imgs, labels, test_size=VAL_SPLIT, random_state=42, stratify=labels)
    print(f"\n  Train {Xtr.shape[0]}   Val {Xv.shape[0]}")
    return Xtr, Xv, ytr, yv


# ─── 3. augmentation ─────────────────────────────────────
def generators(Xtr, ytr, Xv, yv):
    tg = ImageDataGenerator(
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.25, horizontal_flip=True,
        brightness_range=[0.7, 1.3], fill_mode='nearest')
    vg = ImageDataGenerator()
    tg.fit(Xtr); vg.fit(Xv)
    return (tg.flow(Xtr, ytr, batch_size=BATCH_SIZE),
            vg.flow(Xv,  yv,  batch_size=BATCH_SIZE))


# ─── 4. build ─────────────────────────────────────────────
def build():
    print("\n🤖 Building MobileNetV2…")
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(224, 224, 3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_PHASE1),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    tot = model.count_params()
    tra = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Total {tot:,}   Trainable {tra:,}   Frozen {tot-tra:,}")
    return model, base


# ─── 5. train ─────────────────────────────────────────────
def _cbs():
    return [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_OUT, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    ]

def phase1(model, tg, vg):
    print("\n🚀 Phase 1 — head only…")
    return model.fit(tg, validation_data=vg,
                     epochs=EPOCHS_PHASE1, callbacks=_cbs(), verbose=1)

def phase2(model, base, tg, vg):
    print("\n🔧 Phase 2 — fine-tune last 30 layers…")
    for layer in base.layers[-30:]:
        layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_PHASE2),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    tra = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"  Now trainable: {tra:,}")
    return model.fit(tg, validation_data=vg,
                     epochs=EPOCHS_PHASE2, callbacks=_cbs(), verbose=1)


# ─── 6. evaluate ──────────────────────────────────────────
def evaluate(model, Xv, yv):
    print("\n📊 Evaluation…")
    pred  = np.argmax(model.predict(Xv), axis=1)
    truth = np.argmax(yv, axis=1)
    print(classification_report(truth, pred, target_names=GESTURE_CLASSES))
    print("Confusion matrix:\n", confusion_matrix(truth, pred))
    _, acc = model.evaluate(Xv, yv, verbose=0)
    print(f"\n  Val accuracy: {acc*100:.2f} %")
    return acc


# ─── 7. plot ──────────────────────────────────────────────
def plot(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'],     label='Train', color='#4CAF50')
    ax[0].plot(history.history['val_accuracy'], label='Val',   color='#2196F3')
    ax[0].set(title='Accuracy', ylabel='Accuracy', xlabel='Epoch')
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(history.history['loss'],     label='Train', color='#f44336')
    ax[1].plot(history.history['val_loss'], label='Val',   color='#FF9800')
    ax[1].set(title='Loss', ylabel='Loss', xlabel='Epoch')
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("  📈 training_history.png saved")


# ─── main ─────────────────────────────────────────────────
def main():
    print("=" * 58)
    print("   MOBILENET GESTURE CLASSIFIER — TRAINING")
    print("=" * 58)
    t0 = time.time()

    imgs, labels = load_dataset()
    if not len(imgs):
        print("\n❌ No images.  Run capture_images.py first.")
        return

    imgs, labels_oh = preprocess(imgs, labels)
    Xtr, Xv, ytr, yv = split(imgs, labels_oh)
    tg, vg = generators(Xtr, ytr, Xv, yv)
    model, base = build()

    h1 = phase1(model, tg, vg)
    phase2(model, base, tg, vg)

    acc = evaluate(model, Xv, yv)
    plot(h1)

    # save class names
    with open(CLASS_NAMES_OUT, 'w') as f:
        json.dump(GESTURE_CLASSES, f, indent=2)
    print(f"  💾 {CLASS_NAMES_OUT}")

    print("\n" + "=" * 58)
    print("   ✅ DONE")
    print("=" * 58)
    print(f"  Model      : {MODEL_OUT}")
    print(f"  Accuracy   : {acc*100:.2f} %")
    print(f"  Time       : {(time.time()-t0)/60:.1f} min")
    print(f"  Next step  : python gesture_console.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
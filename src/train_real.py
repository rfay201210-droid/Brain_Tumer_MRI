"""
Train real models on the provided Roboflow MRI dataset.
Saves best models to models/ folder.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# sklearn optional – we print simple accuracy if not available
try:
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Paths
TRAIN_DIR = "/tmp/train_raw/train"
VALID_DIR = "/tmp/valid_raw/valid"
TEST_DIR  = "/tmp/test_raw/test"
MODEL_DIR = "/home/workdir/artifacts/brain_tumor_project/models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Class order must match folder names
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

def make_generators():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest"
    )
    val_test = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=True,
        seed=SEED
    )
    val_gen = val_test.flow_from_directory(
        VALID_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False
    )
    test_gen = val_test.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASS_NAMES,
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def build_custom_cnn():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(4, activation="softmax")
    ], name="Custom_CNN")
    return model

def build_mobilenet():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(4, activation="softmax")
    ], name="MobileNetV2")
    return model, base

def train_and_save(model, train_gen, val_gen, name, epochs=EPOCHS, lr=1e-3):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    ckpt = os.path.join(MODEL_DIR, f"{name}_best.h5")
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(ckpt, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cbs,
        verbose=1
    )
    # also save final
    model.save(os.path.join(MODEL_DIR, f"{name}_final.h5"))
    return history, ckpt

def evaluate(model, test_gen, name):
    test_gen.reset()
    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    acc = float(np.mean(y_pred == y_true))
    print(f"\n===== {name} Test Results =====")
    print(f"Accuracy: {acc:.4f}")
    if HAS_SKLEARN:
        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    else:
        for i, cls in enumerate(CLASS_NAMES):
            mask = y_true == i
            if mask.sum() > 0:
                cls_acc = (y_pred[mask] == i).mean()
                print(f"  {cls}: {cls_acc:.3f} ({mask.sum()} samples)")
    return acc

def main():
    print("Creating generators...")
    train_gen, val_gen, test_gen = make_generators()
    print("Class indices:", train_gen.class_indices)

    results = {}

    # 1. Custom CNN
    print("\n" + "="*50 + "\nTraining Custom CNN\n" + "="*50)
    cnn = build_custom_cnn()
    hist, path = train_and_save(cnn, train_gen, val_gen, "custom_cnn", epochs=EPOCHS)
    acc = evaluate(cnn, test_gen, "Custom CNN")
    results["custom_cnn"] = {"accuracy": acc, "path": path}

    # 2. MobileNetV2
    print("\n" + "="*50 + "\nTraining MobileNetV2\n" + "="*50)
    mob, base = build_mobilenet()
    hist, path = train_and_save(mob, train_gen, val_gen, "mobilenetv2", epochs=EPOCHS)
    acc = evaluate(mob, test_gen, "MobileNetV2")
    results["mobilenetv2"] = {"accuracy": acc, "path": path}

    # 3. Fine-tune MobileNet
    print("\n" + "="*50 + "\nFine-tuning MobileNetV2\n" + "="*50)
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    hist, path = train_and_save(mob, train_gen, val_gen, "mobilenetv2_finetuned", epochs=8, lr=1e-4)
    acc = evaluate(mob, test_gen, "MobileNetV2 Fine-tuned")
    results["mobilenetv2_finetuned"] = {"accuracy": acc, "path": path}

    # Save metadata matching app class order
    # App expects: 0=Glioma, 1=Meningioma, 2=No Tumor, 3=Pituitary
    class_indices = train_gen.class_indices  # glioma:0, meningioma:1, no_tumor:2, pituitary:3
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n===== FINAL COMPARISON =====")
    for k, v in results.items():
        print(f"{k}: Test Accuracy = {v['accuracy']:.4f}")
    print("Models saved to", MODEL_DIR)

if __name__ == "__main__":
    main()

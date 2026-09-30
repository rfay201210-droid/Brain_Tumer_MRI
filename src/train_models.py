"""
Brain Tumor MRI Image Classification - Training Script
Trains Custom CNN and Transfer Learning models (MobileNetV2, EfficientNetB0)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Config
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15
NUM_CLASSES = 4
CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

def create_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "Training"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=SEED,
        shuffle=True
    )
    val_gen = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "Training"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        seed=SEED,
        shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "Testing"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return train_gen, val_gen, test_gen

def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="Custom_CNN")
    return model

def build_transfer_model(base_name="mobilenet", input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    if base_name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
        name = "MobileNetV2"
    else:
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        name = "EfficientNetB0"

    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name=name)
    return model, base

def compile_and_train(model, train_gen, val_gen, model_name, epochs=EPOCHS):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    ckpt_path = os.path.join(MODEL_DIR, f"{model_name}_best.h5")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=cbs,
        verbose=1
    )
    # Save final
    final_path = os.path.join(MODEL_DIR, f"{model_name}_final.h5")
    model.save(final_path)
    print(f"Saved {final_path}")
    return history, ckpt_path

def evaluate_model(model, test_gen, model_name):
    test_gen.reset()
    preds = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    report = classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys()), output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n=== {model_name} Evaluation ===")
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))
    return report, cm

def plot_history(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title(f"{model_name} Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name} Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{model_name}_history.png"), dpi=120)
    plt.close()

def main():
    print("Creating data generators...")
    train_gen, val_gen, test_gen = create_generators()
    print("Class indices:", train_gen.class_indices)

    results = {}

    # 1. Custom CNN
    print("\n" + "="*50)
    print("Training Custom CNN")
    print("="*50)
    cnn = build_custom_cnn()
    hist_cnn, path_cnn = compile_and_train(cnn, train_gen, val_gen, "custom_cnn", epochs=EPOCHS)
    plot_history(hist_cnn, "custom_cnn")
    report_cnn, cm_cnn = evaluate_model(cnn, test_gen, "Custom CNN")
    results["custom_cnn"] = {"report": report_cnn, "best_path": path_cnn}

    # 2. MobileNetV2
    print("\n" + "="*50)
    print("Training MobileNetV2 (Transfer Learning)")
    print("="*50)
    mob, base_mob = build_transfer_model("mobilenet")
    hist_mob, path_mob = compile_and_train(mob, train_gen, val_gen, "mobilenetv2", epochs=EPOCHS)
    plot_history(hist_mob, "mobilenetv2")
    report_mob, cm_mob = evaluate_model(mob, test_gen, "MobileNetV2")
    results["mobilenetv2"] = {"report": report_mob, "best_path": path_mob}

    # Fine-tune top layers of MobileNet
    print("\nFine-tuning MobileNetV2 top layers...")
    base_mob.trainable = True
    for layer in base_mob.layers[:-30]:
        layer.trainable = False
    mob.compile(optimizer=optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    hist_ft, path_ft = compile_and_train(mob, train_gen, val_gen, "mobilenetv2_finetuned", epochs=8)
    plot_history(hist_ft, "mobilenetv2_finetuned")
    report_ft, _ = evaluate_model(mob, test_gen, "MobileNetV2 Fine-tuned")
    results["mobilenetv2_finetuned"] = {"report": report_ft, "best_path": path_ft}

    # 3. EfficientNetB0
    print("\n" + "="*50)
    print("Training EfficientNetB0 (Transfer Learning)")
    print("="*50)
    eff, _ = build_transfer_model("efficientnet")
    hist_eff, path_eff = compile_and_train(eff, train_gen, val_gen, "efficientnetb0", epochs=EPOCHS)
    plot_history(hist_eff, "efficientnetb0")
    report_eff, cm_eff = evaluate_model(eff, test_gen, "EfficientNetB0")
    results["efficientnetb0"] = {"report": report_eff, "best_path": path_eff}

    # Save comparison
    comparison = {}
    for name, res in results.items():
        acc = res["report"].get("accuracy", 0)
        comparison[name] = {
            "accuracy": acc,
            "macro_f1": res["report"].get("macro avg", {}).get("f1-score", 0),
            "path": res["best_path"]
        }
    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print("\n=== Model Comparison ===")
    for k, v in comparison.items():
        print(f"{k}: Acc={v['accuracy']:.4f}, Macro-F1={v['macro_f1']:.4f}")

    # Save class indices for Streamlit
    with open(os.path.join(MODEL_DIR, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    print("\nTraining complete. Models saved to:", MODEL_DIR)

if __name__ == "__main__":
    main()

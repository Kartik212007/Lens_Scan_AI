import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os
import json

# =========================
# Paths — all relative to THIS script file (BUG-02 fix)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE   = 128
BATCH_SIZE = 32
EPOCHS     = 30   # raised; EarlyStopping will stop when val_loss stops improving

# BUG-01 FIX: Dataset is double-nested inside Plant_Disease_Dataset/Plant_Disease_Dataset/
train_dir = os.path.join(BASE_DIR, "Plant_Disease_Dataset", "Plant_Disease_Dataset", "train")
valid_dir = os.path.join(BASE_DIR, "Plant_Disease_Dataset", "Plant_Disease_Dataset", "valid")

# Make sure the paths exist before attempting to generate data
if not os.path.isdir(train_dir) or not os.path.isdir(valid_dir):
    raise FileNotFoundError(
        f"Training or validation directory not found.\n"
        f"Expected:\n  {train_dir}\n  {valid_dir}\n"
        "Please download/extract the Plant Disease Dataset and check the folder structure."
    )

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.15,          # extra augmentation for better generalization
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Sanity check: we must have at least one class to train on
if train_generator.num_classes == 0:
    raise RuntimeError("No classes found in training directory. Check that images are organized in subfolders.")

print(f"Found {train_generator.num_classes} classes for training.")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# BUG-05 FIX: Callbacks — Early Stopping + Best-model Checkpoint
# =========================
model_save_path = os.path.join(BASE_DIR, "trained_model.keras")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,           # stop if val_loss doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# BUG-02 FIX: Save model & labels relative to script dir, not CWD
# =========================
# Also save .h5 for legacy compatibility
h5_save_path = os.path.join(BASE_DIR, "trained_model.h5")
model.save(h5_save_path)
print(f"✅ Model saved to: {h5_save_path}")

# Save class label mapping
labels_save_path = os.path.join(BASE_DIR, "class_labels.pkl")
with open(labels_save_path, "wb") as f:
    pickle.dump(train_generator.class_indices, f)
print(f"✅ Class labels saved to: {labels_save_path}")

# Save training history as JSON for the UI to read dynamically
hist_save_path = os.path.join(BASE_DIR, "training_hist.json")
with open(hist_save_path, "w") as f:
    json.dump(history.history, f)
print(f"✅ Training history saved to: {hist_save_path}")

print("\n🎉 Training complete!")
final_val_acc = max(history.history.get("val_accuracy", [0])) * 100
print(f"   Best Val Accuracy: {final_val_acc:.2f}%")

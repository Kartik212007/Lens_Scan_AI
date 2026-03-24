import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import pickle
import os

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

train_dir = "Plant_Disease_Dataset/train"
valid_dir = "Plant_Disease_Dataset/valid"

# make sure the paths exist before attempting to generate data
if not os.path.isdir(train_dir) or not os.path.isdir(valid_dir):
    raise FileNotFoundError(
        f"Training or validation directory not found.\n"
        f"Expected {train_dir} and {valid_dir} to exist.\n"
        "Please download/extract the Plant Disease Dataset and update the paths accordingly."
    )

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
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

# sanity check: we must have at least one class to train on
if train_generator.num_classes == 0:
    raise RuntimeError("No classes found in training directory, check that images are organized in subfolders.")

print(f"Found {train_generator.num_classes} classes for training: {list(train_generator.class_indices.keys())}")

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

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

model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS
)

# Save Model
model.save("trained_model.h5")

# Save Class Order (VERY IMPORTANT)
with open("class_labels.pkl", "wb") as f:
    pickle.dump(train_generator.class_indices, f)

print("✅ Model & Class labels saved successfully!")

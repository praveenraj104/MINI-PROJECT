import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "C:/Users/praveenraj/OneDrive/Documents/MINI PROJECT/archive (1)/PlantVillage"   # Update with your dataset path

# Load dataset using ImageDataGenerator
datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255.0)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())
np.save("class_names.npy", class_names)  # Save class names

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    
    Dense(num_classes, activation="softmax")  # ✅ FIXED: 9 output classes instead of 2
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save the trained model
model.save("plant_disease_model.h5")

print("✅ Model training completed and saved!")

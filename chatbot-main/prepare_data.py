import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Function to preprocess the image (resize, normalize, etc.)
def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for img_path in image_paths:
        # Read the image using OpenCV
        image = cv2.imread(img_path)
        
        # Resize the image to the target size
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values (scale them between 0 and 1)
        image = image / 255.0
        
        # Add the image to the list
        images.append(image)
    
    return np.array(images)

# Define the dataset directory (change this to the path of your dataset)
dataset_dir = "C:/Users/praveenraj/OneDrive/Documents/MINI PROJECT/archive (1)/PlantVillage"   # Replace with your dataset folder path

# Get a list of image paths and labels
image_paths = []
labels = []
for label_name in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label_name)
    if os.path.isdir(label_dir):
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            if image_path.endswith(".jpg") or image_path.endswith(".png"):
                image_paths.append(image_path)
                labels.append(label_name)

# Convert labels to integers
unique_labels = sorted(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels_int = [label_map[label] for label in labels]

# Preprocess images
images = load_and_preprocess_images(image_paths)

# Convert labels to one-hot encoding
labels_one_hot = to_categorical(labels_int, num_classes=len(unique_labels))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Now X_train is defined, you can proceed with model building.

# Define the CNN model architecture
def build_model(input_shape, num_classes):
    model = Sequential()
    
    # Convolutional Layer 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    
    # Convolutional Layer 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Convolutional Layer 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # Flatten the feature map
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))  # For multi-class classification
    
    return model

# Build the model
input_shape = X_train.shape[1:]  # (224, 224, 3) or whatever input shape your images have
num_classes = y_train.shape[1]  # Number of disease classes

model = build_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  # Multi-class classification
              metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,  # Number of epochs
    batch_size=32,  # Batch size
    callbacks=[early_stopping]
)

# Print final accuracy
print("Training Accuracy:", history.history['accuracy'][-1])
print("Validation Accuracy:", history.history['val_accuracy'][-1])

# Evaluate the model on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Save the model
model.save('plant_disease_model.h5')
print("Model saved as 'plant_disease_model.h5'")

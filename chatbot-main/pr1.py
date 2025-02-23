import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define the directory where your dataset is stored
dataset_dir = "C:/Users/praveenraj/OneDrive/Documents/MINI PROJECT/archive (1)/PlantVillage"  # Replace with your dataset path

# Function to load and preprocess images
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

# Get a list of image paths (assuming images are stored in subfolders by label)
image_paths = []
labels = []

# Iterate over the dataset directory and collect image paths and labels
for label_name in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label_name)
    if os.path.isdir(label_dir):
        for image_name in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_name)
            if image_path.endswith(".jpg") or image_path.endswith(".png"):
                image_paths.append(image_path)
                labels.append(label_name)  # Use folder name as label

# Convert labels to integers
unique_labels = sorted(set(labels))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
labels_int = [label_map[label] for label in labels]

# Preprocess images
images = load_and_preprocess_images(image_paths)

# Convert labels to one-hot encoding
labels_one_hot = to_categorical(labels_int, num_classes=len(unique_labels))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

# Save the preprocessed data to disk (optional)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)

print("Data preprocessing complete.")

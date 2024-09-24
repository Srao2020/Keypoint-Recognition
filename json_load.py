import json
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Define file paths
data_dir = '/Users/25rao/Desktop/mlai/Project_1/data/Training'
images_dir = os.path.join(data_dir, 'images')
annotations_dir = os.path.join(data_dir, 'annotations')

# Initialize lists
images = []
keypoints = []

# List all files in the images directory for debugging
print("Images in directory:")
for img_file in os.listdir(images_dir):
    print(img_file)

# Load and process data
for json_file in os.listdir(annotations_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(annotations_dir, json_file)

        # Load keypoints from JSON file
        with open(json_path) as f:
            entry = json.load(f)

        # Construct image filename from the JSON filename
        image_file = json_file.replace('_A.json', '_R.png')  # Change to .png
        image_path = os.path.join(images_dir, image_file)

        keypoint_array = []

        # Navigate through the annotations to find keypoints
        for annotation in entry.get('annotations', []):
            for result in annotation.get('annotations', []):
                for keypoint in result.get('result', []):
                    keypoints_data = keypoint.get('value', {})
                    if 'keypointlabels' in keypoints_data:
                        keypoint_array.append(keypoints_data['x'])
                        keypoint_array.append(keypoints_data['y'])

        if not keypoint_array:
            continue

        # Load and preprocess image
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (512, 512))  # Resize to 512x512
                image = image / 255.0  # Normalize image

                images.append(image)
                keypoints.append(keypoint_array)  # Append the flattened keypoints
        else:
            print(f"Warning: Image not found at {image_path}")

# Convert lists to numpy arrays
images = np.array(images)
keypoints = np.array(keypoints)

# Check if images and keypoints are loaded
if images.size == 0 or keypoints.size == 0:
    print("Error: No images or keypoints loaded. Please check your directories.")
else:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, keypoints, test_size=0.2, random_state=42)
    print(f"Loaded {len(images)} images and {len(keypoints)} keypoints.")

    # Save the datasets as .npy files
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    print("Datasets saved successfully!")
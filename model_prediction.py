import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/Users/25rao/Desktop/mlai/Project_1/data/models/model.h5')

# Define paths
input_folder = '/Users/25rao/Desktop/mlai/Project_1/data/dataset/images'
output_folder = '/Users/25rao/Desktop/mlai/Project_1/data/dataset/output'

def predict_keypoints(image_path):
    image = cv2.imread(image_path)
    #image = cv2.resize(image, (512, 512))  # Resize to match model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    keypoints = model.predict(image)
    return keypoints

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        image_path = os.path.join(input_folder, filename)
        predicted_keypoints = predict_keypoints(image_path)

        # Prepare output data with separate (x, y) coordinates
        output_data = []
        for i in range(len(predicted_keypoints[0]) // 2):  # Assuming each keypoint has (x, y)
            x = predicted_keypoints[0][i * 2]
            y = predicted_keypoints[0][i * 2 + 1]
            output_data.append(f"{x:.2f}, {y:.2f}")  # Format as x, y

        # Save the predicted keypoints to a .txt file
        output_txt_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_keypoints.txt')
        with open(output_txt_path, 'w') as txt_file:
            txt_file.write("\n".join(output_data))  # Write each coordinate pair on a new line

        print(f"Predicted keypoints for {filename} saved to {output_txt_path}")

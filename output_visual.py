import os
import cv2
import matplotlib.pyplot as plt

# Define paths
images_folder = '/Users/25rao/Desktop/mlai/Project_1/data/dataset/images/'
txt_folder = '/Users/25rao/Desktop/mlai/Project_1/data/dataset/output/'

def visualize_keypoints(image_path, keypoints):
    # Read and display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    plt.imshow(image)

    # Plot keypoints with smaller, colored dots
    for idx, point in enumerate(keypoints):
        x, y = point  # Point is a tuple (x, y)
        plt.scatter(x, y, color=plt.cm.jet(idx / len(keypoints)), s=20, marker='o')  # Use colormap and smaller size

    plt.title(os.path.basename(image_path))
    plt.axis('off')
    plt.show()

# Process each .txt file in the output folder
for filename in os.listdir(txt_folder):
    if filename.endswith('_keypoints.txt'):
        txt_path = os.path.join(txt_folder, filename)

        # Load the keypoints from the .txt file
        keypoints = []
        with open(txt_path, 'r') as txt_file:
            for line in txt_file:
                x, y = map(float, line.strip().split(','))  # Convert to float
                keypoints.append((x, y))  # Store as tuples

        # Generate corresponding image path by removing '_keypoints' from the filename
        image_name = os.path.splitext(filename)[0].replace('_keypoints', '') + '.png'  # Adjust format if needed
        image_path = os.path.join(images_folder, image_name)

        # Visualize keypoints if the image exists
        if os.path.exists(image_path):
            visualize_keypoints(image_path, keypoints)
        else:
            print(f"Image not found for {filename}: {image_name}")
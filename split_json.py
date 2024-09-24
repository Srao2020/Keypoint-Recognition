# separate and rename json files

import json
import os

# Define paths (Ensure these paths are absolute and correctly formatted for your system)
annotations_json_path = '/Users/25rao/Desktop/mlai/Project_1/data/Training/annotations_1/project-2-at-2024-09-22-12-38-fa25f0ef.json' # update for new data
images_folder_path = '/Users/25rao/Desktop/mlai/Project_1/dataTraining//images'
output_json_folder_path = '/Users/25rao/Desktop/mlai/Project_1/data/Training/annotations'

# Ensure the output folder exists
os.makedirs(output_json_folder_path, exist_ok=True)

# Load the original annotations JSON file
with open(annotations_json_path, 'r') as f:
    annotations_data = json.load(f)

# Verify that annotations_data is a list
if not isinstance(annotations_data, list):
    raise ValueError("The loaded JSON data is not a list. Please check the structure of your JSON file.")

# Get the list of images from the folder and sort them
image_files = sorted(
    [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])

# Create a dictionary to store annotations by image file name
annotations_by_image = {}

# Extract image names from annotations and organize them
for annotation in annotations_data:
    file_upload = annotation.get('file_upload', '')
    if not file_upload:
        continue

    # Extract the image name from the file_upload field
    image_name = os.path.splitext(file_upload.split('-')[-1])[0]
    if image_name not in annotations_by_image:
        annotations_by_image[image_name] = []

    # Add the entire annotation to the list for this image
    annotations_by_image[image_name].append(annotation)

# Save each image's annotations into a separate JSON file in the output folder
for image_file in image_files:
    image_name, _ = os.path.splitext(image_file)
    if image_name in annotations_by_image:
        image_annotations = annotations_by_image[image_name]
        output_json_path = os.path.join(output_json_folder_path, f'{image_name}.json')

        # Write the image annotations to a new JSON file
        with open(output_json_path, 'w') as f:
            json.dump({'annotations': image_annotations}, f, indent=4)

        print(f'Saved annotations for {image_file} to {output_json_path}')
    else:
        print(f'No annotations found for {image_file}')

print('Processing complete.')

# change file names

images_folder_path = '/Users/25rao/Desktop/mlai/Project_1/data/Training/annotations'

# List all files in the images folder
image_files = [f for f in os.listdir(images_folder_path) if f.lower().endswith(('.json'))]

# Process each file and rename if it contains '_R'
for image_file in image_files:
    # Check if the file name contains '_R'
    if '_R' in image_file:
        # Construct the new file name by replacing '_R' with '_A'
        new_image_file = image_file.replace('_R', '_A')

        # Define the full paths for renaming
        old_file_path = os.path.join(images_folder_path, image_file)
        new_file_path = os.path.join(images_folder_path, new_image_file)

        # Rename the file
        os.rename(old_file_path, new_file_path)

        print(f'Renamed: {old_file_path} -> {new_file_path}')

print('Renaming complete.')
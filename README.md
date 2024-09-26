# Project 1 MLAI Keypoint Model

# Why I built this model / future work

This model was built in order to map key points on images of feet wearing flip flops. Mapping the key points will allow the model to eventually quantify toe spread the distance between the first two toes. 

Diabetes can cause a loss of sensation in the lower extremities a condition called diabetic neuropathy and quality footwear is incredibly important to prevent complications such as diabetic foot ulcers. Over a 100 million people globally have diabetic neuropathy 18 million of whom develop diabetic foot ulcers and 6 million of them will get amputations (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7919962/). A study found that the pressure from traditional straight edged flip flop straps is responsible for as much as 15% of diabetic foot ulcers caused by footwear (https://pubmed.ncbi.nlm.nih.gov/29916424/). 

5 years ago I designed a new type of flip flop strap called the rolled inner seam strap. The purpose was to reduce localized pressure and risk of abrasion on the top of the foot specifically the diabetic foot. 
In areas where the only affordable footwear is flip flops and access to good healthcare is low this innovation could be pivotal. 

I wanted to have evidence that the rolled inner seam flip flops did in fact reduce localized pressure. So I conducted human walking trials. I placed a pressure sensor at the toehold area and had my volunteers walk in my designed flip flops and a pretty commonly worn flip flop. After analyzing my results I found that there was a statistically significant difference and the rolled inner seam reduced the pressure. 

Then I started thinking about the actual workings of the foot. Since I was measuring pressure at the toe hold area the distance between those two toes would play into the measurement. So I wanted to find if there was any correlation between toe spread type of flip flop and pressure. The end goal/product of this project is a software in which you can take a picture of your foot while wearing a flip flop feed it into my model tell it the type of flip flop you are wearing and it will tell you the estimated pressure that flip flop will have on your foot. 

# Data Preprocessing

My data comes from videos I took while conducting my walking trials. I selected frames and resized them to be 512 by 512 pixels. I used ImageJ (https://imagej.net/ij/). Since I wanted to calculate the distance between the two toes in pixels I needed to make sure the scale of the images was the same. I used the strap of my flip flop which is around 1.25 cm and used pixel ratios to scale the images. I did that for 78 images. After resizing them I uploaded my raw dataset to Label Studio (https://labelstud.io/) and annotated them with two keypoints. This software exports the dataset in one json file so I used the first script split_json.py to separate each image's annotations. This code uses the image filenames as keys to organize and map the annotations. It also renames JSON files in a folder by replacing specific substrings in the filenames utilizing Python's os and json modules for file handling directory management and data manipulation. Then I used json_load.py to load and process the annotations. The script imports json for handling annotation files os for managing file paths and directories numpy for array manipulations cv2 from OpenCV for reading resizing and normalizing images and train_test_split from sklearn.model_selection to split the dataset into training and testing sets. The images were further processed and saved as NumPy arrays for use in machine learning. It loads images from a specified directory normalizes them by using the standard divide by 255 and extracts keypoint coordinates from corresponding JSON annotation files. The images and keypoints are stored in lists which are later converted into NumPy arrays. After ensuring data is properly loaded the script splits the data into training and testing sets using train_test_split. Finally it saves these datasets as .npy files for future use ensuring the data is ready for machine learning tasks such as keypoint detection or object recognition.

- Split JSON annotations using `split_json.py`.
- Process images and annotations using `json_load.py`.

# Model 

After all of the preprocessing steps I ran model_train.py. This code builds a convolutional neural network (CNN) for keypoint detection from images. I decided to use this because similar projects used this format and network and its functionality fit my purpose. I used NumPy for efficient handling and loading of large datasets ensuring compatibility with TensorFlow which provides the necessary deep learning framework. The model architecture built using TensorFlow’s Keras API leverages convolutional layers (Conv2D) with increasing filters (32 64 128) to progressively extract features from the images which is essential for identifying patterns in spatial data. MaxPooling layers follow each convolution to downsample the feature maps reducing computational complexity and preventing overfitting. The Flatten layer transforms the 2D feature maps into a 1D vector for the fully connected (Dense) layers where 512 neurons (using ReLU activation) capture complex feature relationships followed by 4 output neurons for the predicted keypoint coordinates. Adam optimizer is used for its efficient learning rate adaptation while mean squared error (MSE) is chosen as the loss function appropriate for the continuous nature of keypoint predictions in a regression task. The model is trained for 10 epochs with a batch size of 32 using a 10% validation split to monitor performance on unseen data and reduce overfitting risks. Finally the trained model is saved allowing for future use without retraining ensuring reproducibility and deployment readiness. I used a virtual environment in Pycharm. I had initially decided to use vscode but it was way more complicated. 
- Train the model using `model_train.py`.
- **Conv2D layers**: To extract features from images.
- **MaxPooling layers**: To downsample feature maps.
- **Dense layers**: To capture complex feature relationships.
- **Output**: 4 keypoint coordinates.

# Model Testing

To test the model and evaluate its performance I used model_test.py. This code script evaluates my pre-trained convolutional neural network by loading test data and the trained model. The use of NumPy ensures efficient handling of the test dataset which is necessary for TensorFlow’s input format. The preprocessed test data (X_test and y_test) are loaded from .npy files which is both memory efficient and fast for handling large datasets in array format. The trained model is then loaded using TensorFlow’s load_model function which allows the script to retrieve the saved model architecture and learned weights from a previously trained instance ensuring the evaluation is performed on the exact model that was trained. The model is evaluated on the test dataset using the evaluate method which computes the loss based on the mean squared error (MSE) loss function used during training. This loss metric is printed to provide insight into how well the model generalizes to unseen data a key part of the model validation process. Overall the choices in this script were optimized for efficiency and reproducibility and to ensure reliable performance evaluation. In evaluating the loss metric this script outputs I first trained the model using a dataset size of 30 and then 78. The lower the loss metric the better. I found that when I used 30 images the metric was 822.3. And with 78 images it was 370.3. So it can be concluded that with a larger dataset the model will be more accurate. However due to time constraints I wasn’t able to for this project. In the near future I will create a larger training dataset. 

- Test the model using `model_test.py`.

# Model Predictions/Results

To implement my model I used model_prediction.py. This script processes the images predicts keypoints using the model and saves the results to text files. The script begins by loading the required libraries: os for directory traversal cv2 (OpenCV) for image reading and preprocessing and numpy for efficient numerical operations. The trained model is loaded using load_model from TensorFlow’s Keras API ensuring that the pre-trained weights and architecture are correctly restored for inference. The image files are read from the specified input_folder using OpenCV which is a widely-used library for image processing ensuring seamless reading and manipulation of various image formats. The predict_keypoints function processes each image by normalizing the pixel values (dividing by 255.0 to scale between 0 and 1). The image is reshaped with np.expand_dims to add a batch dimension making it compatible with the model's expected input format. The model predicts the keypoints which are the coordinates representing specific features of the image. For each image the predicted keypoints are parsed into (x y) pairs assuming that the model outputs an even number of values where every two consecutive values represent a keypoint’s x and y coordinates. These coordinates are formatted as two decimal precision strings for clarity. The results are then saved to a .txt file in the output_folder ensuring the predicted keypoints for each image are easily accessible for later analysis or visualization. The script is optimized for handling multiple images in a batch as it iterates through all image files in the input directory allowing for efficient keypoint prediction on large datasets. The use of os.path ensures compatibility across operating systems when handling file paths. 

- Predict keypoints with `model_prediction.py`.

# Steps to run model

**Root Directory**

- Data
    - Training
        - Annotations_1 *the full json file*
        - annotations *the split json file*
        - images *the raw training data*
    - Dataset
        - images *the dataset*
        - output
    - Models *This is where the model is saved*

- Update the file paths in each of the files to match yours
- Run `split_json.py` if your files are not separated by image yet
- Run `json_load.py`
- Run `model_train.py`
- Run `model_test.py`
- Run `model_prediction.py`

# Project Collaboration
- Label Studio - for image annotations
- ImageJ - pixel ratios and resizing
- PycharmPro - the IDE I used
- ChatGPT - used for errors and debugging

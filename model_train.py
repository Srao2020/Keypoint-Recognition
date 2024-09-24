import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load preprocessed data
X_train = np.load('/Users/25rao/Desktop/mlai/Project_1/data/Training/X_train.npy')
y_train = np.load('/Users/25rao/Desktop/mlai/Project_1/data/Training/y_train.npy', allow_pickle=True)
X_test = np.load('/Users/25rao/Desktop/mlai/Project_1/data/Training/X_test.npy')
y_test = np.load('/Users/25rao/Desktop/mlai/Project_1/data/Training/y_test.npy', allow_pickle=True)

# Check types and shapes
print(type(y_train), y_train.shape)

# Convert y_train to a NumPy array if necessary
if isinstance(y_train, dict):
    y_train = np.array(list(y_train.values()))  # Adjust as needed

# Check the shape again
print(y_train.shape)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(4)  # Adjusted to match the shape of y_train (2 keypoints: x1, y1, x2, y2)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

# Save the trained model
model.save('/Users/25rao/Desktop/mlai/Project_1/data/models/model.h5')

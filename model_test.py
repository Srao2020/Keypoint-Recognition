# Model Evaluation Script
import numpy as np
from tensorflow.keras.models import load_model

# Load preprocessed test data
X_test = np.load('/Users/25rao/Desktop/mlai/Project_1/data/Training/X_test.npy')
y_test = np.load('/Users/25rao/Desktop/mlai/Project_1/data/Training/y_test.npy')

# Load the trained model
model = load_model('/Users/25rao/Desktop/mlai/Project_1/data/models/model.h5')

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
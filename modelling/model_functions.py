# Required imports
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# Defining the majority pooling function for the baseline model
# TO DO - fix
def majority_pool(array, new_shape=(2, 2, 6)):
    # Initialize the new pooled array
    pooled_array = np.zeros(new_shape, dtype=array.dtype)

    # Calculate the step sizes for x and y dimensions
    step_x = array.shape[0] // new_shape[0]
    step_y = array.shape[1] // new_shape[1]

    # Loop through each dimension
    for z in range(array.shape[2]):
        for new_x in range(new_shape[0]):
            for new_y in range(new_shape[1]):
                x_start = new_x * step_x
                y_start = new_y * step_y
                quadrant = array[x_start:x_start+step_x, y_start:y_start+step_y, z]
                majority = np.sum(quadrant) > (step_x * step_y // 2)
                pooled_array[new_x, new_y, z] = majority

    return pooled_array

# Define the baseline model
def baseline(test_feature):
# TO DO - add docstring
 # TO DO - fix
    mask_ndvi = NDVI >= 0.6
    NDVI_bucketed = np.where(mask_ndvi, 1, 0)
    NDVI_pooled = majority_pool(NDVI_bucketed)

# Define and train the CNN model
def train_cnn():
    # TO DO - Ana / Shayan your code here
    pass

# Run a prediction from the CNN model
def predict_cnn():
    # TO DO - Ana / Shayan your code here
    pass

# Define the evaluation function
def evaluate(test_target, y_pred):
    """ Evaluates the model using the F1 score. """
    # Reshape your arrays into 1D arrays
    true_values_1D = test_target.reshape(-1)
    pred_values_1D = y_pred.reshape(-1)

    # Calculate F1 score
    f1 = f1_score(true_values_1D, pred_values_1D)

    print("F1 Score:", f1)

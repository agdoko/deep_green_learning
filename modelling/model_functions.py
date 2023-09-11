# Required imports
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

# Defining the majority pooling function for the baseline model
def majority_pool(array):
    dt = np.dtype([("ndvi", np.int32)])
    pooled_array = np.zeros((array.shape[0], 3, 3), dtype=dt)

    step_x = step_y = 16

    for n in range(array.shape[0]):
        for i in range(3):
            x_start = i * step_x
            for j in range(3):
                y_start = j * step_y
                quadrant = array[n, x_start:x_start+step_x, y_start:y_start+step_y]
                majority = np.sum(quadrant) > (step_x * step_y // 2)
                pooled_array[n, i, j]['ndvi'] = int(majority)

    return pooled_array

# Define the baseline model
def baseline(test_feature):
    """ Baseline model that uses majority pooling on a threshold NDVI value. """
    mask_ndvi = test_feature >= 0.6
    NDVI_bucketed = np.where(mask_ndvi, 1, 0)
    y_pred_baseline = majority_pool(NDVI_bucketed)
    return y_pred_baseline

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
    #true_values_1D = test_target.reshape(-1)
    #pred_values_1D = y_pred.reshape(-1)

    true_values_1D = test_target["landcover"].flatten()
    pred_values_1D = y_pred['ndvi'].flatten()

    # Calculate F1 score
    f1 = f1_score(true_values_1D, pred_values_1D)

    print("F1 Score:", f1)

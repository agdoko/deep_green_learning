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

#Convert a structured ndarray to a standard ndarray and expand dimensions to align with CNN input.
def process_and_expand(structured_array, field_name, dtype=np.float32):
    """
    Parameters:
        structured_array (numpy.ndarray): The structured array to convert.
        field_name (str): The name of the field to extract from the structured array.
        dtype (numpy.dtype, optional): The desired dtype for the output array. Defaults to np.float32.

    Returns:
        numpy.ndarray: The converted and expanded standard ndarray.
    """
    standard_array = np.array(structured_array[field_name], dtype=dtype)
    expanded_array = np.expand_dims(standard_array, axis=-1)
    return expanded_array


# Define and train the CNN model
def train_cnn(expanded_array):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_feature_expanded, train_target_expanded, epochs=10, batch_size=32, validation_split=0.2)

    return model

trained_model = train_cnn(train_feature_reshaped, train_target_reshaped)

# Run a prediction from the CNN model
def predict_cnn(model, test_feature):
    predictions = model.predict(test_feature)
    # Round predictions to get binary classification output
    rounded_predictions = [round(x[0]) for x in predictions]
    return rounded_predictions

predictions = predict_cnn(trained_model, test_feature_reshaped)
print("Predictions:", predictions)

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

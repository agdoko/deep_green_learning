# Required imports
import numpy as np
from tensorflow import keras
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import Sequential, layers

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.metrics import Metric
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from tensorflow.keras.utils import model_to_dot, plot_model, image_dataset_from_directory
#from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Reshape

from sklearn.metrics import f1_score

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def majority_pool(array):
    dt = np.dtype([("ndvi", np.int32)])
    pooled_array = np.zeros((array.shape[0], 1, 1), dtype=dt)

    for n in range(array.shape[0]):
        total = np.sum(array[n])
        majority = total > ((50 * 50) // 2)
        pooled_array[n, 0, 0]['ndvi'] = int(majority)

    return pooled_array


# # CHECKPOINTING WORKING MAJORITY POOLING FUNCTION BUT LEGACY 3x3 format
# # Defining the majority pooling function for the baseline model
# def majority_pool(array):
#     dt = np.dtype([("ndvi", np.int32)])
#     pooled_array = np.zeros((array.shape[0], 3, 3), dtype=dt)

#     step_x = step_y = 16

#     for n in range(array.shape[0]):
#         for i in range(3):
#             x_start = i * step_x
#             for j in range(3):
#                 y_start = j * step_y
#                 quadrant = array[n, x_start:x_start+step_x, y_start:y_start+step_y]
#                 majority = np.sum(quadrant) > (step_x * step_y // 2)
#                 pooled_array[n, i, j]['ndvi'] = int(majority)

#     return pooled_array

# Define the baseline model
def baseline(test_feature):
    """ Baseline model that uses majority pooling on a threshold NDVI value. """
    mask_ndvi = test_feature >= 0.6
    NDVI_bucketed = np.where(mask_ndvi, 1, 0)
    y_pred_baseline = majority_pool(NDVI_bucketed)
    return y_pred_baseline

#Convert a structured ndarray to a standard ndarray and expand dimensions to align with CNN input.
def process_and_expand(structured_array,  dtype=np.float32):
    """
    Parameters:
        structured_array (numpy.ndarray): The structured array to convert.
        field_name (str): The name of the field to extract from the structured array.
        dtype (numpy.dtype, optional): The desired dtype for the output array. Defaults to np.float32.

    Returns:
        numpy.ndarray: The converted and expanded standard ndarray.
    """
    standard_array = np.array(structured_array, dtype=dtype)
    expanded_array = np.expand_dims(standard_array, axis=-1)
    return expanded_array

# Define and train the CNN model
def train_cnn(processed_train_f, processed_train_t):
    input_img = Input(shape=(50, 50, 1))

    # Convolutional Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    # Convolutional Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    # Convolutional Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(0.25)(x)

    # Flattening the feature map
    x = Flatten()(x)

    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    # Dense layer to get the desired number of features
    x = Dense(1*1*1, activation='sigmoid')(x)

    # Reshaping to the target shape
    decoded = Reshape((1, 1, 1))(x)

    # Create the model
    model = tensorflow.keras.models.Model(input_img, decoded)

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=[F1Score()])

    # Fit the model
    history = model.fit(
        processed_train_f,
        processed_train_t,
        epochs=100,
        batch_size=32,
        shuffle=True
    )
    plot = f1_plot(history)
    #plot.savefig
    return model

# Run a prediction from the CNN model
def predict_with_model(model, processed_test_f):
    y_pred = model.predict(processed_test_f)
    return y_pred

# Define the evaluation function
def evaluate_model(model, processed_test_f, processed_test_t):
    loss, f1_score = model.evaluate(processed_test_f, processed_test_t)
    print("Test Loss:", loss)
    print("Test F1 Score:", f1_score)
    return loss, f1_score

def f1_plot(history):

    # Extract F1 scores from the training history
    f1_scores = history.history['f1_score']  # Adjust 'f1_score' if your metric has a different name

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the F1 scores
    ax.plot(f1_scores, label='F1 Score')

    # Add labels and legend
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Over Epochs')
    ax.legend()

    # Optionally, display the plot if needed
    plt.show()

    # Return the figure object
    return fig

#def predict_cnn(model, test_feature):
    #predictions = model.predict(test_feature)
    # Round predictions to get binary classification output
    #rounded_predictions = [round(x[0]) for x in predictions]
    #return rounded_predictions

# predictions = predict_cnn(trained_model, test_feature_reshaped)
# print("Predictions:", predictions)

# Define the evaluation function
#def evaluate(test_target, y_pred):
    """ Evaluates the model using the F1 score. """
    # Reshape your arrays into 1D arrays
    #true_values_1D = test_target.reshape(-1)
    #pred_values_1D = y_pred.reshape(-1)

#    true_values_1D = test_target["landcover"].flatten()
#    pred_values_1D = y_pred['ndvi'].flatten()

#    # Calculate F1 score
#    f1 = f1_score(true_values_1D, pred_values_1D)

#    print("F1 Score:", f1)

#F1Score
class F1Score(Metric):

    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tensorflow.round(y_pred)
        values = tensorflow.cast(y_true, tensorflow.float32)
        predictions = tensorflow.cast(y_pred, tensorflow.float32)
        self.true_positives.assign_add(tensorflow.reduce_sum(values * predictions))
        self.false_positives.assign_add(tensorflow.reduce_sum((1.0 - values) * predictions))
        self.false_negatives.assign_add(tensorflow.reduce_sum(values * (1.0 - predictions)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tensorflow.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tensorflow.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tensorflow.keras.backend.epsilon()))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import load_model

from src.preprocessing.data_management import load_dataset

# Custom callback to print loss after each epoch
class PrintLossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch # {epoch + 1}, Loss = {logs['loss']}")

def preprocess_data(X, y):
    # Normalize or scale your data if needed
    X_normalized = (X - X.mean()) / X.std()
    y_normalized = (y - y.mean()) / y.std()
    return X_normalized, y_normalized

def build_model(input_dim, num_layers, layer_units, activation_functions):
    model = Sequential()
    for i in range(num_layers):
        if i == 0:
            model.add(Dense(layer_units[i], input_dim=input_dim, activation=activation_functions[i]))
        else:
            model.add(Dense(layer_units[i], activation=activation_functions[i]))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    return model

def run_training(tol, epsilon, mini_batch_size=2, epochs=2, patience=10):
    # Load and preprocess data
    training_data = load_dataset("train.csv")
    X_train, Y_train = preprocess_data(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    # Define model parameters
    input_dim = X_train.shape[1]
    num_layers = 2  # Example: input layer, one hidden layer
    layer_units = [10, 10]  # Example: [hidden units, hidden units]
    activation_functions = ['relu', 'relu']  # Example activations

    # Build and compile the model
    model = build_model(input_dim, num_layers, layer_units, activation_functions)
    optimizer = Adam(learning_rate=epsilon)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='loss', min_delta=tol, patience=patience, verbose=1, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=mini_batch_size,
        verbose=0,  # Turn off default verbose output
        callbacks=[PrintLossCallback(), early_stopping]
    )

    # Save the model
    model.save("two_input_xor_nn_keras.h5")

    return history.history['loss']

if __name__ == "__main__":
    run_training(10**(-8), 10**(-7), mini_batch_size=2)

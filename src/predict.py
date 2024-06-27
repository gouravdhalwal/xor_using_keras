import numpy as np
from tensorflow.keras.models import load_model

def load_trained_model():
    model = load_model("two_input_xor_nn_keras.h5")  # Load your Keras model
    return model

def predict(input_data):
    model = load_trained_model()
    predictions = model.predict(input_data)
    predictions_prob = 1 / (1 + np.exp(-predictions))  # Applying sigmoid to convert logits to probabilities
    return predictions_prob

def calculate_accuracy(predictions, expected_outputs):
    correct_count = 0
    for i, prediction in enumerate(predictions):
        predicted_class = 1 if prediction >= 0.5 else 0
        if predicted_class == expected_outputs[i]:
            correct_count += 1
    accuracy = correct_count / len(expected_outputs) * 100
    return accuracy

if __name__ == "__main__":
    input_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    expected_outputs = np.array([0, 1, 1, 0])  # Expected outputs for XOR function

    predictions = predict(input_data)
    accuracy = calculate_accuracy(predictions, expected_outputs)

    for i, pred in enumerate(predictions):
        print(f"Prediction probabilities for input {input_data[i]}: {pred[0]}")

    print(f"Accuracy: {accuracy:.2f}%")

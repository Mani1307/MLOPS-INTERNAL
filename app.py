
import pickle
import numpy as np

# Load the already trained model
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

def predict_diabetes(features):
    """
    Predict diabetes using the trained model.
    Features must contain 8 input values.
    """

    if len(features) != 8:
        raise ValueError("Exactly 8 features are required")

    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)

    return int(prediction[0])


if __name__ == "__main__":
    sample_data = [
        6, 148, 72, 35, 0, 33.6, 0.627, 50
    ]

    result = predict_diabetes(sample_data)

    if result == 1:
        print("Diabetes detected")
    else:
        print("No diabetes detected")

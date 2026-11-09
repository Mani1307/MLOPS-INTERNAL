
from app import predict_diabetes

def test_prediction():
    sample_data = [
        6, 148, 72, 35, 0, 33.6, 0.627, 50
    ]

    result = predict_diabetes(sample_data)

    assert result in [0, 1]


def test_invalid_input():
    sample_data = [1, 2, 3]

    try:
        predict_diabetes(sample_data)
    except ValueError:
        return

    assert False, "Expected ValueError for invalid input"

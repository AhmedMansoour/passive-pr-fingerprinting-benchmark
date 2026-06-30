import numpy as np

from src.data_loader import load_dataset
from src.models.kwnn import train_kwnn
from src.evaluation import compute_euclidean_errors


def test_kwnn_smoke_prediction_shape():
    X_train, y_train, X_test, y_test, _ = load_dataset("data/raw")
    model = train_kwnn(X_train, y_train, k=3, metric="euclidean")
    pred = model.predict(X_test)
    errors = compute_euclidean_errors(y_test, pred)
    assert pred.shape == y_test.shape == (9, 2)
    assert errors.shape == (9,)
    assert np.isfinite(errors).all()

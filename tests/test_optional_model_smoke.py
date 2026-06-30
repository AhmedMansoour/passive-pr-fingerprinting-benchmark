import importlib.util

import numpy as np
import pytest

from src.data_loader import load_dataset


def test_sklearn_random_forest_smoke():
    from src.models.random_forest import train_random_forest
    from src.evaluation import compute_euclidean_errors

    X_train, y_train, X_test, y_test, _ = load_dataset("data/raw", scale=True)
    model = train_random_forest(X_train, y_train, n_estimators=5, random_state=42, n_jobs=1)
    pred = model.predict(X_test)
    err = compute_euclidean_errors(y_test, pred)
    assert pred.shape == (9, 2)
    assert np.isfinite(err).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_pytorch_mlp_and_transformer_forward_smoke():
    import torch
    from src.models.improved_models_pytorch import ImprovedMLP, ImprovedTransformer

    x = torch.randn(4, 6)
    mlp = ImprovedMLP(input_dim=6, hidden_layers=[8, 4], dropout_rate=0.0)
    tr = ImprovedTransformer(input_dim=6, d_model=8, num_heads=2, num_layers=1, ff_dim=16, dropout=0.0)
    assert mlp(x).shape == (4, 2)
    assert tr(x).shape == (4, 2)

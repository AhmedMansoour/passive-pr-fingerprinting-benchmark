"""Multi-Layer Perceptron models (scikit-learn) for indoor localization."""

from sklearn.neural_network import MLPRegressor


MLP_ARCHITECTURES = {
    "MLP_Arch1_3Layer": (128, 64, 32),
    "MLP_Arch2_4Layer": (256, 128, 64, 32),
    "MLP_Arch3_Tapered": (256, 128, 64),
    "MLP_Arch4_Wide": (512, 256, 128),
}


def train_mlp_sklearn(X_train, y_train, hidden_layers=(128, 64, 32), random_state=42):
    """Train an MLP regressor using scikit-learn."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        max_iter=40,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,
        learning_rate_init=0.001,
        alpha=0.0001,
        batch_size=16,
        activation="relu",
        tol=1e-3,
        n_iter_no_change=6,
    )
    model.fit(X_train, y_train)
    return model


def run_mlp_multiseed(X_train, y_train, X_test, arch_name, hidden_layers, seeds):
    """Run MLP with multiple seeds, return list of (seed, y_pred) pairs."""
    results = []
    for seed in seeds:
        model = train_mlp_sklearn(X_train, y_train, hidden_layers, random_state=seed)
        y_pred = model.predict(X_test)
        results.append({"seed": seed, "y_pred": y_pred, "model": model})
    return results

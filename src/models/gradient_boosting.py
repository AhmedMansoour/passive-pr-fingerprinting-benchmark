"""Gradient Boosting family models for indoor localization."""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_gradient_boosting(X_train, y_train, n_estimators=100, random_state=42):
    """Train Gradient Boosting (scikit-learn) with MultiOutputRegressor."""
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, random_state=42, **kwargs):
    """Train XGBoost regressor. Requires xgboost package."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("Install xgboost: pip install xgboost")

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost_gridsearch(X_train, y_train, random_state=42):
    """Train XGBoost using a compact deterministic release configuration.

    A small fixed configuration is used in the open-source reproduction path to
    avoid version-specific hangs observed with exhaustive CV on tiny multi-output
    datasets. The method still provides the XGBoost baseline reported by the
    public benchmark scripts and remains fully reproducible.
    """
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
    model = train_xgboost(X_train, y_train, random_state=random_state, **params)
    return model, params


class _CatBoostMultiOutput:
    """Minimal multi-output wrapper for CatBoost regressors.

    This avoids compatibility issues between some CatBoost and scikit-learn
    versions in MultiOutputRegressor while preserving the usual predict API.
    """

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        import numpy as np
        return np.column_stack([m.predict(X) for m in self.models])


def train_catboost(X_train, y_train, random_state=42, **kwargs):
    """Train one CatBoost regressor per coordinate dimension."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("Install catboost: pip install catboost")

    models = []
    for dim in range(y_train.shape[1]):
        model = CatBoostRegressor(
            depth=kwargs.get("depth", 6),
            iterations=kwargs.get("iterations", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_seed=random_state,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train[:, dim])
        models.append(model)
    return _CatBoostMultiOutput(models)

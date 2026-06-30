"""Random Forest regression for indoor localization."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42, n_jobs=-1):
    """Train a Random Forest regressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_gridsearch(X_train, y_train, random_state=42):
    """Train Random Forest with grid search over hyperparameters."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    gs = GridSearchCV(base, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_

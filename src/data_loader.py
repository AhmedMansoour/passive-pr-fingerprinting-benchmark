"""Shared data loading and preprocessing for WiFi fingerprint localization."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.experiment_config import (
    DEFAULT_RSSI, CSV_SEPARATOR, TRAIN_COORD_MAPPING, TEST_COORD_MAPPING
)


def load_raw_data(data_dir):
    """Load raw CSV files."""
    train_raw = pd.read_csv(os.path.join(data_dir, "df_all.csv"), sep=CSV_SEPARATOR)
    test_raw = pd.read_csv(os.path.join(data_dir, "test_df.csv"), sep=CSV_SEPARATOR)
    return train_raw, test_raw


def build_fingerprint_matrix(train_raw, test_raw):
    """
    Build fingerprint matrices from raw probe request data.

    Steps:
    1. Lowercase point/folder names
    2. Aggregate RSSI by (Point, AP) → mean signal
    3. Pivot to create feature matrix (rows=points, cols=APs)
    4. Fill missing APs with DEFAULT_RSSI (-100 dBm)
    5. Map to ground-truth coordinates
    """
    # Training data
    train_raw["Point"] = train_raw["Point"].str.lower()
    train_agg = train_raw.groupby(["Point", "AP"]).agg({"Signal": "mean"}).reset_index()
    train_pivot = train_agg.pivot(index="Point", columns="AP", values="Signal").fillna(DEFAULT_RSSI).reset_index()
    train_pivot["X"] = train_pivot["Point"].apply(lambda p: TRAIN_COORD_MAPPING.get(p, (np.nan, np.nan))[0])
    train_pivot["Y"] = train_pivot["Point"].apply(lambda p: TRAIN_COORD_MAPPING.get(p, (np.nan, np.nan))[1])
    train_pivot = train_pivot.dropna()

    # Test data
    test_raw["Folder"] = test_raw["Folder"].str.lower()
    test_agg = test_raw.groupby(["Folder", "AP"]).agg({"Signal": "mean"}).reset_index()
    test_pivot = test_agg.pivot(index="Folder", columns="AP", values="Signal").fillna(DEFAULT_RSSI).reset_index()
    test_pivot["X"] = test_pivot["Folder"].apply(lambda f: TEST_COORD_MAPPING.get(f, (np.nan, np.nan))[0])
    test_pivot["Y"] = test_pivot["Folder"].apply(lambda f: TEST_COORD_MAPPING.get(f, (np.nan, np.nan))[1])
    test_pivot = test_pivot.dropna()

    return train_pivot, test_pivot


def extract_features_and_targets(train_pivot, test_pivot, standardize=False):
    """
    Extract feature matrices X and target matrices y.

    Returns X_train, y_train, X_test, y_test, ap_columns, scaler (or None).
    Coordinates are in millimeters.
    """
    ap_columns = sorted([col for col in train_pivot.columns if str(col).lower().startswith("ap")])

    X_train = train_pivot[ap_columns].values
    y_train = train_pivot[["X", "Y"]].values
    X_test = test_pivot[ap_columns].values
    y_test = test_pivot[["X", "Y"]].values

    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, ap_columns, scaler


def load_dataset(train_csv, test_csv=None, coord_map_train=None, coord_map_test=None, scale=False):
    """
    Complete data loading pipeline.

    Args:
        train_csv: Path to training CSV, or a directory containing df_all.csv and test_df.csv.
        test_csv: Path to test CSV. If None, train_csv is treated as a directory.
        coord_map_train: Training coordinate mapping. Defaults to TRAIN_COORD_MAPPING.
        coord_map_test: Test coordinate mapping. Defaults to TEST_COORD_MAPPING.
        scale: Whether to standardize features.

    Returns: (X_train, y_train, X_test, y_test, scaler)
    """
    if test_csv is None:
        train_raw, test_raw = load_raw_data(train_csv)
    else:
        train_raw = pd.read_csv(train_csv, sep=CSV_SEPARATOR)
        test_raw = pd.read_csv(test_csv, sep=CSV_SEPARATOR)

    _train_map = coord_map_train if coord_map_train is not None else TRAIN_COORD_MAPPING
    _test_map = coord_map_test if coord_map_test is not None else TEST_COORD_MAPPING

    train_raw["Point"] = train_raw["Point"].str.lower()
    train_agg = train_raw.groupby(["Point", "AP"]).agg({"Signal": "mean"}).reset_index()
    train_pivot = train_agg.pivot(index="Point", columns="AP", values="Signal").fillna(DEFAULT_RSSI).reset_index()
    train_pivot["X"] = train_pivot["Point"].apply(lambda p: _train_map.get(p, (np.nan, np.nan))[0])
    train_pivot["Y"] = train_pivot["Point"].apply(lambda p: _train_map.get(p, (np.nan, np.nan))[1])
    train_pivot = train_pivot.dropna()

    test_raw["Folder"] = test_raw["Folder"].str.lower()
    test_agg = test_raw.groupby(["Folder", "AP"]).agg({"Signal": "mean"}).reset_index()
    test_pivot = test_agg.pivot(index="Folder", columns="AP", values="Signal").fillna(DEFAULT_RSSI).reset_index()
    test_pivot["X"] = test_pivot["Folder"].apply(lambda f: _test_map.get(f, (np.nan, np.nan))[0])
    test_pivot["Y"] = test_pivot["Folder"].apply(lambda f: _test_map.get(f, (np.nan, np.nan))[1])
    test_pivot = test_pivot.dropna()

    X_train, y_train, X_test, y_test, _, scaler = extract_features_and_targets(
        train_pivot, test_pivot, standardize=scale
    )
    return X_train, y_train, X_test, y_test, scaler

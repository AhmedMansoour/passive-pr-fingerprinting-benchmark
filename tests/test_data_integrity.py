from pathlib import Path

import pandas as pd

from configs.experiment_config import TRAIN_COORD_MAPPING, TEST_COORD_MAPPING

ROOT = Path(__file__).resolve().parents[1]


def test_raw_data_shape_privacy_and_mapping_consistency():
    train = pd.read_csv(ROOT / "data" / "raw" / "df_all.csv", sep=";")
    test = pd.read_csv(ROOT / "data" / "raw" / "test_df.csv", sep=";")

    assert train["Point"].nunique() == 36
    assert test["Folder"].nunique() == 9
    assert train["AP"].nunique() == 6
    assert test["AP"].nunique() == 6

    assert set(train["Point"].str.lower().unique()) == set(TRAIN_COORD_MAPPING)
    assert set(test["Folder"].str.lower().unique()) == set(TEST_COORD_MAPPING)

    assert "MAC Address" not in train.columns
    assert "MAC Address" not in test.columns
    assert "Device_ID" in train.columns
    assert "Device_ID" in test.columns


def test_processed_fingerprints():
    train = pd.read_csv(ROOT / "data" / "processed" / "train_fingerprints.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test_fingerprints.csv")
    ap_cols = [f"ap{i}" for i in range(1, 7)]

    assert len(train) == 36
    assert len(test) == 9
    assert set(["location_id", "X_mm", "Y_mm", *ap_cols]).issubset(train.columns)
    assert set(["location_id", "X_mm", "Y_mm", *ap_cols]).issubset(test.columns)

"""Coordinate mapping definitions for the indoor localization testbed."""

from configs.experiment_config import TRAIN_COORD_MAPPING, TEST_COORD_MAPPING

# Total reference points
NUM_TRAIN_POINTS = len(TRAIN_COORD_MAPPING)  # 36
NUM_TEST_POINTS = len(TEST_COORD_MAPPING)    # 9

# Study area bounds (mm)
X_MIN = 0
X_MAX = max(x for x, y in list(TRAIN_COORD_MAPPING.values()) + list(TEST_COORD_MAPPING.values()))
Y_MIN = 0
Y_MAX = max(y for x, y in list(TRAIN_COORD_MAPPING.values()) + list(TEST_COORD_MAPPING.values()))

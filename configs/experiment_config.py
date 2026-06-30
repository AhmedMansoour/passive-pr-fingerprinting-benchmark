"""
Experiment configuration for WiFi fingerprint indoor localization benchmark.
All hyperparameters, coordinate mappings, and experimental settings.
"""

import numpy as np

# =============================================================================
# DATA
# =============================================================================
DEFAULT_RSSI = -100  # dBm, used for missing AP values
CSV_SEPARATOR = ";"

# =============================================================================
# COORDINATE MAPPINGS (millimeters)
# =============================================================================
TRAIN_COORD_MAPPING = {
    **{f"a{i}": (2400 * (i // 2), 0) for i in range(1, 24, 2)},
    **{f"c{i}": (2400 * (i // 2), 2400) for i in range(1, 24, 2)},
    **{f"e{i}": (2400 * (i // 2) + 50, 6080) for i in range(1, 24, 2)},
}

TEST_COORD_MAPPING = {
    "tb3": (2400, 1200),
    "tb7": (7200, 1200),
    "tb10": (10800, 1200),
    "tb14": (15600, 1200),
    "td7": (7200, 3600),
    "td11": (12000, 3600),
    "td18": (20400, 3600),
    "td20": (22800, 3600),
    "td22": (25200, 3600),
}

# =============================================================================
# MULTI-SEED REPRODUCIBILITY
# =============================================================================
RANDOM_SEEDS = [42, 123, 456, 789, 999]
NUM_SEEDS = len(RANDOM_SEEDS)

# =============================================================================
# KWNN CONFIGURATIONS
# =============================================================================
KWNN_K_VALUES = [3, 4, 5, 6]
KWNN_DISTANCE_METRICS = ["euclidean", "manhattan", "chebyshev", "minkowski"]

# =============================================================================
# TREE-BASED ENSEMBLE CONFIGURATIONS
# =============================================================================
RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "n_jobs": -1,
}

GRADIENT_BOOSTING_CONFIG = {
    "n_estimators": 100,
}

XGBOOST_PARAM_GRID = {
    "n_estimators": [50, 100],
    "max_depth": [3, 6],
    "learning_rate": [0.01, 0.1],
}

CATBOOST_CONFIG = {
    "depth_range": range(4, 11),
    "iterations_range": range(50, 1001, 50),
    "learning_rate": 0.1,
    "early_stopping_rounds": 100,
    "validation_fraction": 0.1,
}

# =============================================================================
# MLP CONFIGURATIONS (sklearn)
# =============================================================================
MLP_ARCHITECTURES_SKLEARN = {
    "MLP_Arch1_3Layer": (128, 64, 32),
    "MLP_Arch2_4Layer": (256, 128, 64, 32),
    "MLP_Arch3_Tapered": (256, 128, 64),
    "MLP_Arch4_Wide": (512, 256, 128),
}

MLP_SKLEARN_PARAMS = {
    "max_iter": 500,
    "early_stopping": True,
    "validation_fraction": 0.2,
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "batch_size": 16,
    "activation": "relu",
}

# =============================================================================
# MLP CONFIGURATIONS (TensorFlow/Keras)
# =============================================================================
MLP_ARCHITECTURES_TF = {
    "MLP_Arch1_3Layer": [128, 64, 32],
    "MLP_Arch2_4Layer": [256, 128, 64, 32],
    "MLP_Arch3_Tapered": [256, 128, 64],
    "MLP_Arch4_Wide": [512, 256, 128],
}

MLP_TF_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "activation": "elu",
}

# =============================================================================
# TRANSFORMER CONFIGURATIONS (TensorFlow/Keras)
# =============================================================================
TRANSFORMER_CONFIGS_TF = {
    "Transformer_TF1": {"d_model": 64, "num_heads": 4, "num_layers": 2},
    "Transformer_TF2": {"d_model": 128, "num_heads": 8, "num_layers": 3},
    "Transformer_TF3": {"d_model": 256, "num_heads": 8, "num_layers": 4},
}

TRANSFORMER_TF_PARAMS = {
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2,
    "learning_rate": 0.001,
}

# =============================================================================
# IMPROVED MLP CONFIGURATIONS (PyTorch)
# =============================================================================
IMPROVED_MLP_CONFIGS = {
    "ANN_Improved_S": [128, 64, 32],
    "ANN_Improved_M": [256, 128, 64, 32],
    "ANN_Improved_L": [512, 256, 128, 64],
    "ANN_Improved_XL": [1024, 512, 256, 128, 64],
}

# =============================================================================
# IMPROVED TRANSFORMER CONFIGURATIONS (PyTorch)
# =============================================================================
IMPROVED_TRANSFORMER_CONFIGS = {
    "Transformer_Improved_S": {"d_model": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
    "Transformer_Improved_M": {"d_model": 128, "num_heads": 8, "num_layers": 3, "ff_dim": 512},
    "Transformer_Improved_L": {"d_model": 256, "num_heads": 8, "num_layers": 4, "ff_dim": 1024},
    "Transformer_Improved_XL": {"d_model": 512, "num_heads": 16, "num_layers": 4, "ff_dim": 2048},
}

PYTORCH_TRAINING_PARAMS = {
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "patience": 8,
    "dropout_rate": 0.2,
    "grad_clip": 1.0,
}

# =============================================================================
# LATENCY MEASUREMENT
# =============================================================================
LATENCY_CONFIG = {
    "num_repetitions": 100,
    "warmup_reps": 10,
}

# Fast verification settings used by smoke tests and CI-style release checks.
QUICK_RANDOM_SEEDS = [42]
QUICK_PYTORCH_TRAINING_PARAMS = {
    "epochs": 5,
    "batch_size": 16,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "patience": 3,
    "dropout_rate": 0.2,
    "grad_clip": 1.0,
}

# =============================================================================
# BOOTSTRAP CI
# =============================================================================
BOOTSTRAP_CONFIG = {
    "num_samples": 10000,
    "confidence_level": 0.95,
    "random_seed": 42,
}

# =============================================================================
# AP PHYSICAL POSITIONS (meters, from deployment layout xyaps.png)
# =============================================================================
AP_POSITIONS_M = {
    "ap1": (24.0, 8.0),
    "ap2": (22.0, 5.0),
    "ap3": (12.0, 5.0),
    "ap4": (12.0, 7.0),
    "ap5": (3.0, 1.0),
    "ap6": (7.0, 5.0),
}

# =============================================================================
# DENSITY SENSITIVITY ANALYSIS
# =============================================================================
DENSITY_ANALYSIS_SEEDS = [42, 123, 456]

"""Multi-Layer Perceptron models (TensorFlow/Keras) for indoor localization."""

import numpy as np


def create_mlp_model(layer_sizes, input_dim=6, dropout_rate=0.2, activation="elu", lr=0.001):
    """Create a Keras MLP model."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers

    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for i, units in enumerate(layer_sizes):
        model.add(layers.Dense(units, activation=activation))
        if i < len(layer_sizes) - 1:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(2))
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def train_mlp_tf(X_train, y_train, layer_sizes, seed=42, epochs=100, batch_size=32,
                 validation_split=0.2, input_dim=6, dropout_rate=0.2,
                 activation="elu", learning_rate=0.001):
    """Train a Keras MLP with a specific random seed."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = create_mlp_model(layer_sizes, input_dim=input_dim, dropout_rate=dropout_rate, activation=activation, lr=learning_rate)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split, verbose=0)
    return model


MLP_ARCHITECTURES_TF = {
    "MLP_Arch1_3Layer": [128, 64, 32],
    "MLP_Arch2_4Layer": [256, 128, 64, 32],
    "MLP_Arch3_Tapered": [256, 128, 64],
    "MLP_Arch4_Wide": [512, 256, 128],
}

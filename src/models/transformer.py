"""Transformer-based models (TensorFlow/Keras) for indoor localization."""

import numpy as np


def create_transformer_model(d_model, num_heads, num_layers, input_dim=6, lr=0.001):
    """Create a simplified Transformer encoder model."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, optimizers

    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(d_model)(inputs)
    x = layers.Reshape((1, d_model))(x)

    for _ in range(num_layers):
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization()(x)

        ff = layers.Dense(d_model * 4, activation="relu")(x)
        ff = layers.Dense(d_model)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(2)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def train_transformer_tf(X_train, y_train, config, seed=42, epochs=100, batch_size=32,
                         validation_split=0.2, input_dim=6, learning_rate=0.001):
    """Train a Transformer model with a specific seed."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = create_transformer_model(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        input_dim=input_dim,
        lr=learning_rate,
    )
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split, verbose=0)
    return model


TRANSFORMER_CONFIGS = {
    "Transformer_TF1": {"d_model": 64, "num_heads": 4, "num_layers": 2},
    "Transformer_TF2": {"d_model": 128, "num_heads": 8, "num_layers": 3},
    "Transformer_TF3": {"d_model": 256, "num_heads": 8, "num_layers": 4},
}

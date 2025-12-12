from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


class ModelBuilder:
    """Builds a simple feedforward classifier for Hyrox specialty."""

    def __init__(self, input_dim: int, num_classes: int, hidden_units: int = 32, dropout: float = 0.3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.dropout = dropout

    def build(self) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(self.hidden_units, activation="relu"),
            layers.Dropout(self.dropout),
            layers.Dense(self.hidden_units, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

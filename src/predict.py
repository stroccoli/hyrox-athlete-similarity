from typing import List

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .utils import load_artifact, model_paths


class Predictor:
    """Loads artifacts and produces predictions given raw feature inputs."""

    def __init__(self, name: str = "hyrox_specialty"):
        paths = model_paths(name=name)
        self.scaler = load_artifact(paths["scaler"])  # type: ignore
        self.label_encoder = load_artifact(paths["label_encoder"])
        self.model = load_model(paths["model"])

    def predict_from_dataframe(self, df: pd.DataFrame, feature_columns: List[str]) -> List[str]:
        X = df[feature_columns].fillna(df[feature_columns].mean())
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        classes = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(classes).tolist()

    def predict_from_array(self, X: np.ndarray) -> List[str]:
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict(X_scaled)
        classes = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(classes).tolist()

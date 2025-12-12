from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class Schema:
    run_columns: List[str]
    workstation_columns: List[str]

    @property
    def feature_columns(self) -> List[str]:
        return self.run_columns + self.workstation_columns


class Preprocessor:
    """Handles input preprocessing: scaling features and encoding labels."""

    def __init__(self, schema: Schema):
        self.schema = schema
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def fit(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        X = df[self.schema.feature_columns].fillna(df[self.schema.feature_columns].mean())
        y = df[target_col]
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        return X_scaled, y_encoded

    def transform_features(self, df_or_array: pd.DataFrame) -> np.ndarray:
        if isinstance(df_or_array, pd.DataFrame):
            X = df_or_array[self.schema.feature_columns].fillna(df_or_array[self.schema.feature_columns].mean())
            return self.scaler.transform(X)
        return self.scaler.transform(df_or_array)

    def decode_labels(self, y_pred_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_pred_encoded)

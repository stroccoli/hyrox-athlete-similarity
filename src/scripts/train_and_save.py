from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import Schema, Preprocessor
from src.model import ModelBuilder
from src.utils import save_artifact, save_json, model_paths


def main():
    df = pd.read_csv("data/hyrox_elite15_synthetic_dataset.csv")

    schema = Schema(
        run_columns=['run1', 'run2', 'run3', 'run4', 'run5', 'run6', 'run7', 'run8'],
        workstation_columns=['ski', 'sled_push', 'sled_pull', 'burpees', 'row', 'farmers', 'lunges', 'wallballs'],
    )

    pre = Preprocessor(schema)
    X_scaled, y_encoded = pre.fit(df, target_col="specialty")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    builder = ModelBuilder(input_dim=X_scaled.shape[1], num_classes=len(pre.label_encoder.classes_))
    model = builder.build()

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=20, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    paths = model_paths()
    save_artifact(pre.scaler, paths["scaler"])
    save_artifact(pre.label_encoder, paths["label_encoder"])
    model.save(paths["model"])
    save_json({
        "feature_columns": schema.feature_columns,
        "classes": list(pre.label_encoder.classes_),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
    }, paths["metadata"])


if __name__ == "__main__":
    main()

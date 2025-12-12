import argparse
from pathlib import Path

import pandas as pd

from src.predict import Predictor
from src.utils import load_json, model_paths


def main():
    parser = argparse.ArgumentParser(description="Predict Hyrox athlete specialty from CSV")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV containing feature columns")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    meta = load_json(model_paths()["metadata"])
    feature_columns = meta["feature_columns"]

    predictor = Predictor()
    labels = predictor.predict_from_dataframe(df, feature_columns)

    out_path = Path(args.input).with_suffix(".predictions.csv")
    pd.DataFrame({"prediction": labels}).to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()

from pathlib import Path
import json
from typing import Any, Dict, List, Optional

import joblib

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_MODEL_DIR.mkdir(exist_ok=True)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_artifact(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path) -> Any:
    return joblib.load(path)


def model_paths(base_dir: Path = DEFAULT_MODEL_DIR, name: str = "hyrox_specialty") -> Dict[str, Path]:
    return {
        "scaler": base_dir / f"{name}_scaler.joblib",
        "label_encoder": base_dir / f"{name}_label_encoder.joblib",
        "model": base_dir / f"{name}_model.keras",
        "metadata": base_dir / f"{name}_metadata.json",
    }

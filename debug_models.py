#!/usr/bin/env python3
# Debug script to check model file contents.

import joblib
from pathlib import Path


def check_model_file() -> None:
    model_file = Path("models/aiaa.joblib")

    if not model_file.exists():
        print(f"Model file does not exist: {model_file}")
        return

    try:
        print(f"Loading model file: {model_file}")
        model_data = joblib.load(model_file)

        print(f"Model dataset type: {type(model_data)}")
        print(f"Model dataset keys: {list(model_data.keys()) if isinstance(model_data, dict) else 'Not a dict'}")

        if isinstance(model_data, dict):
            for key, value in model_data.items():
                print(f"  {key}: {type(value)}")
                if key == "models" and isinstance(value, dict):
                    print(f"    Models: {list(value.keys())}")

    except Exception as e:
        print(f"Error loading model file: {e}")
        print(f"Error type: {type(e)}")


if __name__ == "__main__":
    check_model_file()

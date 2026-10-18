"""Predict the failure risk of one production cycle with the trained model.

Example (from the project root):
    python -m src.predict --type L --air 300 --process 310 --rpm 1300 --torque 65 --wear 200
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import add_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "failure_model.joblib"


def predict_risk(product_type, air_temp_k, process_temp_k, rpm, torque_nm, tool_wear_min,
                 model_path: Path = MODEL_PATH):
    """Return (failure probability, alert flag) for one set of sensor readings."""
    bundle = joblib.load(model_path)
    row = pd.DataFrame([{
        "product_type": product_type,
        "air_temp_k": air_temp_k,
        "process_temp_k": process_temp_k,
        "rpm": rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }])
    proba = float(bundle["pipeline"].predict_proba(add_features(row))[0, 1])
    return proba, proba >= bundle["threshold"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", choices=["L", "M", "H"], required=True)
    parser.add_argument("--air", type=float, required=True, help="air temperature [K]")
    parser.add_argument("--process", type=float, required=True, help="process temperature [K]")
    parser.add_argument("--rpm", type=float, required=True, help="rotational speed [rpm]")
    parser.add_argument("--torque", type=float, required=True, help="torque [Nm]")
    parser.add_argument("--wear", type=float, required=True, help="tool wear [min]")
    a = parser.parse_args()

    proba, alert = predict_risk(a.type, a.air, a.process, a.rpm, a.torque, a.wear)
    print(f"Failure probability: {proba:.1%}")
    print("ALERT: schedule a preventive inspection" if alert else "OK: no action needed")


if __name__ == "__main__":
    main()

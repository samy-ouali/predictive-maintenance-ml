"""Feature engineering based on the physics of the machine.

Each engineered feature targets one known failure mechanism:
- temp_diff_k  -> heat dissipation failure (process barely hotter than the air)
- power_w      -> power failure (power too low or too high)
- strain       -> overstrain failure (tool wear x torque too high)
"""

import math

import pandas as pd

NUMERIC_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
    "strain",
]
CATEGORICAL_FEATURES = ["product_type"]
TARGET = "failure"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the engineered features added."""
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    # Mechanical power P = torque x angular speed, with angular speed in rad/s.
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * math.pi / 60
    out["strain"] = out["tool_wear_min"] * out["torque_nm"]
    return out

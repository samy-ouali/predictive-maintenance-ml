import math

import pandas as pd
import pytest

from src.features import add_features


def make_row(**overrides):
    row = {"air_temp_k": 300.0, "process_temp_k": 310.0, "rpm": 1500.0,
           "torque_nm": 40.0, "tool_wear_min": 100.0, "product_type": "M"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_temperature_difference():
    out = add_features(make_row(air_temp_k=298.0, process_temp_k=306.5))
    assert out.loc[0, "temp_diff_k"] == pytest.approx(8.5)


def test_power_uses_angular_speed_in_rad_per_s():
    out = add_features(make_row(rpm=60.0, torque_nm=10.0))
    # 60 rpm = 1 turn per second = 2*pi rad/s, so P = 10 * 2*pi
    assert out.loc[0, "power_w"] == pytest.approx(10 * 2 * math.pi)


def test_strain_is_wear_times_torque():
    out = add_features(make_row(tool_wear_min=200.0, torque_nm=50.0))
    assert out.loc[0, "strain"] == pytest.approx(10_000)


def test_input_is_not_modified():
    df = make_row()
    add_features(df)
    assert "power_w" not in df.columns

"""Load and clean the AI4I 2020 Predictive Maintenance dataset."""

from pathlib import Path
import urllib.request
import zipfile
import io

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ai4i2020.csv"

# Short, code-friendly column names.
RENAME = {
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
    "Type": "product_type",
}

# Columns that describe *how* the machine failed. They are only known after a
# failure, so using them as inputs would be cheating (data leakage).
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def download_data(path: Path = DATA_PATH) -> Path:
    """Download the CSV from the UCI repository if it is not already present."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open("ai4i2020.csv") as src, open(path, "wb") as dst:
            dst.write(src.read())
    return path


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return the cleaned dataset (one row per production cycle)."""
    download_data(path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=RENAME)
    # UDI and Product ID are identifiers, not information about the machine.
    df = df.drop(columns=["UDI", "Product ID"])
    return df

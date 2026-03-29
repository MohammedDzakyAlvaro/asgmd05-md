import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# =========================
# Add project path
# =========================
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import *

# =========================
# Ingest Data
# =========================
def ingest_data() -> None:
    """Copy raw CSV to ingested folder"""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = DATA_RAW_DIR / "train.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"{raw_file} not found. Please add the dataset.")

    df = pd.read_csv(raw_file, sep=",")
    assert not df.empty, "Dataset is empty"

    out_file = DATA_ING_DIR / "train.csv"
    df.to_csv(out_file, sep=",", index=False)
    print(f"Data ingested: {raw_file} → {out_file}")


# =========================
# Load DataFrame
# =========================
def load_frame() -> pd.DataFrame:
    """Load ingested CSV and encode target"""
    path = DATA_ING_DIR / "train.csv"
    df = pd.read_csv(path, sep=",")
    df[TARGET_COL] = df[TARGET_COL].map({True: 1, False: 0})
    return df


# =========================
# Split Features & Target
# =========================
def split_features_target(df: pd.DataFrame):
    """Return X and y"""
    X = df.drop(DROP_COLS + [TARGET_COL], axis=1)
    y = df[TARGET_COL]
    return X, y


# =========================
# Train/Test Split
# =========================
def split_train_test(X: pd.DataFrame, y: pd.Series):
    """Split data into train and test sets"""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
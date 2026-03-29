from pathlib import Path

# =========================
# Base & Data Directories
# =========================
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_RAW_DIR  = BASE_DIR / "data" / "raw"
DATA_ING_DIR  = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# =========================
# Artifacts
# =========================
ARTIFACT_PIPELINE = ARTIFACTS_DIR / "transported_spaceship_pipeline.pkl"

# =========================
# Target & Drop Columns
# =========================
TARGET_COL = "Transported"
DROP_COLS  = ["PassengerId", "Name"]

# =========================
# Features
# =========================
NUM_FEATURES = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_FEATURES = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]

# =========================
# Train/Test Config
# =========================
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# =========================
# Model Parameters
# =========================
MODEL_PARAMS = {'max_iter': 1000}
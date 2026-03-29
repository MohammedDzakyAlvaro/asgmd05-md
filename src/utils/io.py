from pathlib import Path
import joblib

# =========================
# Save & Load Artifacts
# =========================
def save_artifact(obj, path: Path) -> None:
    """Save object to disk with joblib"""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    """Load object from disk, raise error if missing"""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run training pipeline first."
        )
    return joblib.load(path)
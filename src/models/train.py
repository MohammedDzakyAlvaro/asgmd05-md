import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline

from config.config import ARTIFACT_PIPELINE

# =========================
# Train & Save Pipeline
# =========================
def train_pipeline(pipeline: Pipeline, x_train, y_train) -> str:
    """Fit pipeline and save to artifact path"""
    
    # Fit model
    pipeline.fit(x_train, y_train)

    # Ensure artifact directory exists
    ARTIFACT_PIPELINE.parent.mkdir(parents=True, exist_ok=True)

    # Save pipeline
    with open(ARTIFACT_PIPELINE, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Pipeline trained & saved to {ARTIFACT_PIPELINE}")

    return str(ARTIFACT_PIPELINE)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from config.config import MODEL_PARAMS
from src.features.preprocessor import build_transported_preprocessor

# =========================
# Build Full Pipeline
# =========================
def build_pipeline(num_features: list, cat_features: list) -> Pipeline:
    """Create preprocessing + LogisticRegression pipeline"""
    
    # Preprocessing
    preprocessor = build_transported_preprocessor(num_features, cat_features)

    # Pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(**MODEL_PARAMS)),
    ])

    return pipeline
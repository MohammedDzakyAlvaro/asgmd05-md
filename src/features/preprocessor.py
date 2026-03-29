from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =========================
# Build Preprocessor
# =========================
def build_transported_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    """Construct preprocessing pipeline for numeric and categorical features"""

    # =========================
    # Numerical pipeline
    # =========================
    numeric_pipeline = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    # =========================
    # Categorical pipeline
    # =========================
    categorical_pipeline = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # =========================
    # Combine pipelines
    # =========================
    preprocessor = ColumnTransformer(
        transformers=[
            ("numPreprocess", numeric_pipeline, num_features),
            ("catPreprocess", categorical_pipeline, cat_features),
        ],
        remainder="drop",
    )

    return preprocessor
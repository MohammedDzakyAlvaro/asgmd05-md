from src.pipelines.sklearn_pipeline import build_pipeline
from src.models.train import train_pipeline
from src.models.evaluate import evaluate

import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# Main Script
# =========================
def main():
    print("Spaceship Titanic – sklearn pipeline")
    print("=" * 50)

    # =========================
    # Step 1: Load Data
    # =========================
    print("\nStep 1: Load Data")
    df = pd.read_csv("data/raw/train.csv")

    # =========================
    # Step 2: Preprocessing
    # =========================
    print("\nStep 2: Preprocessing")
    df = df.drop(columns=["PassengerId", "Name", "Cabin"])

    X = df.drop(columns=["Transported"])
    y = df["Transported"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # Step 3: Build & Train Pipeline
    # =========================
    print("\nStep 3: Build and Train Pipeline")
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    pipeline = build_pipeline(num_features, cat_features)
    run_id = train_pipeline(pipeline, x_train, y_train)

    # =========================
    # Step 4: Evaluation
    # =========================
    print("\nStep 4: Evaluation")
    accuracy, precision, recall = evaluate(x_test, y_test, run_id)


if __name__ == "__main__":
    main()
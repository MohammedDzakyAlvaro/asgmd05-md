import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =========================
# Evaluate Model
# =========================
def evaluate(x_test, y_test, model_path):
    """Evaluate model performance on test data"""
    
    # Load trained pipeline
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Predictions
    preds = model.predict(x_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    print(f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    return acc, prec, rec
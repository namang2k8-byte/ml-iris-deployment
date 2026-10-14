"""Train, evaluate, and serialize the Iris classification pipeline."""
from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    data = load_iris()
    y = data.target_names[data.target]
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, target_names=data.target_names,
                                   output_dict=True, zero_division=0)
    model_path = MODEL_DIR / "iris_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    metrics = {
        "dataset": "scikit-learn Iris (150 rows, 4 numeric features, 3 classes)",
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "model": "StandardScaler + LogisticRegression",
        "test_accuracy": round(float(accuracy), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": report,
        "feature_names": data.feature_names,
        "class_names": data.target_names.tolist(),
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {MODEL_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()

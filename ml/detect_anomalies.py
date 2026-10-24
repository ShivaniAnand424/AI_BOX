
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / (
    "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
)
MODEL_FILE = BASE_DIR / "isolation_forest.joblib"
RESULT_FILE = BASE_DIR / "isolation_forest_results.json"

# Use numeric input features only; exclude the target label.
TARGET = "ADHERENCE"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    # Isolation Forest is unsupervised: do not train it on ADHERENCE.
    X = df.drop(columns=[TARGET]).select_dtypes(include="number").copy()
    X = X.replace([float("inf"), -float("inf")], float("nan"))
    X = X.fillna(X.median()).fillna(0)

    if len(X) < 10:
        raise ValueError("Not enough records for anomaly detection.")

    X_train, X_test = train_test_split(
        X, test_size=0.20, random_state=42
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # -1 means anomaly; 1 means normal.
    predictions = model.predict(X_test)
    result = {
        "algorithm": "Isolation Forest",
        "records_tested": int(len(X_test)),
        "anomalies_detected": int((predictions == -1).sum()),
        "normal_records": int((predictions == 1).sum()),
        "contamination_setting": 0.05,
        "note": (
            "Unsupervised anomaly detection on numeric dataset features. "
            "An anomaly is not proof of medication non-adherence or a "
            "clinical problem."
        ),
    }

    joblib.dump(
        {"model": model, "features": X.columns.tolist()},
        MODEL_FILE,
    )
    RESULT_FILE.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved results: {RESULT_FILE}")


if __name__ == "__main__":
    main()

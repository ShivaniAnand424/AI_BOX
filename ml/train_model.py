
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / (
    "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
)
MODEL_FILE = BASE_DIR / "adherence_model.joblib"
METRICS_FILE = BASE_DIR / "model_metrics.json"

TARGET = "ADHERENCE"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_excel(DATA_FILE)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = df.dropna(subset=[TARGET]).copy()

    # Encode the target: ADHERENT = 1, NON-ADHERENT = 0.
    y = df[TARGET].map({
        "ADHERENT": 1,
        "NON-ADHERENT": 0,
    })

    if y.isna().any():
        unknown = df.loc[y.isna(), TARGET].unique().tolist()
        raise ValueError(f"Unexpected adherence labels: {unknown}")

    # Exclude the target from the inputs.
    X = df.drop(columns=[TARGET])

    # Convert Boolean features to integers for consistent model input.
    X = X.copy()
    for column in X.select_dtypes(include=["bool"]).columns:
        X[column] = X[column].astype(int)

    # Keep the same feature order for training and future predictions.
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["NON-ADHERENT", "ADHERENT"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "dataset": DATA_FILE.name,
        "total_records": int(len(df)),
        "training_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "features": feature_columns,
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "confusion_matrix_labels": ["NON-ADHERENT", "ADHERENT"],
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": report,
        "note": (
            "Offline evaluation on a random held-out split. "
            "Not clinically validated and not a dose-timing model."
        ),
    }

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
            "target_mapping": {
                "NON-ADHERENT": 0,
                "ADHERENT": 1,
            },
        },
        MODEL_FILE,
    )

    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nMODEL TRAINED SUCCESSFULLY")
    print("-" * 45)
    print(f"Total records:       {len(df)}")
    print(f"Training records:    {len(X_train)}")
    print(f"Test records:        {len(X_test)}")
    print(f"Input features:      {len(feature_columns)}")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Balanced accuracy:   {balanced_accuracy:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["NON-ADHERENT", "ADHERENT"],
        zero_division=0,
    ))

    print("CONFUSION MATRIX")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print(f"\nSaved model:   {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")


if __name__ == "__main__":
    main()

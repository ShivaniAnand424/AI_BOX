
from pathlib import Path
import pandas as pd

DATA_FILE = (
    Path(__file__).parent
    / "data"
    / "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
)

if not DATA_FILE.exists():
    print("ERROR: Dataset file not found!")
    print("Expected location:", DATA_FILE)
    raise SystemExit(1)

print("Dataset found:", DATA_FILE.name)

try:
    excel = pd.ExcelFile(DATA_FILE)

    print("\nSHEETS IN WORKBOOK:")
    for sheet in excel.sheet_names:
        print(f"\n{'=' * 60}")
        print("SHEET:", sheet)

        df = pd.read_excel(DATA_FILE, sheet_name=sheet)

        print("Rows:", len(df))
        print("Columns:", len(df.columns))
        print("\nColumn names:")
        for column in df.columns:
            print(f"- {column}")

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

        if "ADHERENCE" in df.columns:
            print("\nADHERENCE value counts:")
            print(df["ADHERENCE"].value_counts(dropna=False).to_string())

except Exception as error:
    print("\nCould not inspect dataset:", error)

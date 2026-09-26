from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

# Get all CSV files except our previously generated merged dataset
files = sorted(
    f for f in RAW_DIR.glob("*.csv")
    if f.name != "properties_raw.csv"
)

print("=" * 80)
print("RAW PROPERTY FILE VERIFICATION")
print("=" * 80)

print(f"\nCSV files found: {len(files)}")

total_rows = 0

for i, file in enumerate(files, start=1):

    df = pd.read_csv(file)

    total_rows += len(df)

    cities = (
        df["city"].dropna().astype(str).unique().tolist()
        if "city" in df.columns else []
    )

    deal_types = (
        df["dealType"].dropna().astype(str).unique().tolist()
        if "dealType" in df.columns else []
    )

    categories = (
        df["category"].dropna().astype(str).unique().tolist()
        if "category" in df.columns else []
    )

    print(f"\n{i}. {file.name}")
    print(f"   Rows: {len(df)}")
    print(f"   City: {cities}")
    print(f"   Deal type: {deal_types}")
    print(f"   Category: {categories}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Individual CSV files: {len(files)}")
print(f"Total rows: {total_rows}")
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

files = sorted(
    RAW_DIR.glob("dataset_zameen-property-scraper_*.csv")
)

frames = []

for file in files:
    df = pd.read_csv(file)

    df["source_file"] = file.name

    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

missing = combined[combined["listingId"].isna()]

print("=" * 70)
print("MISSING LISTING IDs")
print("=" * 70)

print(f"\nTotal rows: {len(combined)}")
print(f"Missing listingId: {len(missing)}")

if len(missing) > 0:
    columns = [
        "city",
        "dealType",
        "category",
        "propertyTitle",
        "price",
        "locationText",
        "url",
        "source_file"
    ]

    print("\nRecords with missing listingId:\n")
    print(missing[columns].to_string(index=False))

print("\n" + "=" * 70)
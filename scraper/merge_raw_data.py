from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "properties_raw.csv"

files = sorted(
    RAW_DIR.glob("dataset_zameen-property-scraper_*.csv")
)

print("=" * 70)
print("MERGING RAW PROPERTY DATA")
print("=" * 70)

print(f"\nApify CSV files found: {len(files)}")

if len(files) != 27:
    raise ValueError(
        f"Expected 27 Apify CSV files, but found {len(files)}."
    )

frames = []

for file in files:
    df = pd.read_csv(file)

    # Preserve the original source file.
    df["source_file"] = file.name

    # Create a unique record ID.
    # Normal listing IDs use the scraper's listingId.
    # Missing listing IDs use the row number within that source file.
    df["source_listing_id"] = (
        df["listingId"]
        .astype("Int64")
        .astype(str)
    )

    missing_id = df["listingId"].isna()

    df.loc[missing_id, "source_listing_id"] = (
        "row_"
        + df.index[missing_id].astype(str)
    )

    df["source_listing_id"] = (
        df["source_file"].str.replace(".csv", "", regex=False)
        + "_"
        + df["source_listing_id"]
    )

    frames.append(df)

combined = pd.concat(frames, ignore_index=True)

print(f"\nRows before duplicate removal: {len(combined)}")

duplicates = combined.duplicated(
    subset=["source_listing_id"],
    keep="first"
).sum()

print(f"Duplicate source records removed: {duplicates}")

combined = combined.drop_duplicates(
    subset=["source_listing_id"],
    keep="first"
)

print(f"Final rows: {len(combined)}")

print("\n" + "=" * 70)
print("CITY DISTRIBUTION")
print("=" * 70)
print(combined["city"].value_counts())

print("\n" + "=" * 70)
print("DEAL TYPE")
print("=" * 70)
print(combined["dealType"].value_counts())

print("\n" + "=" * 70)
print("CATEGORY")
print("=" * 70)
print(combined["category"].value_counts())

combined.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)
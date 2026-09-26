from pathlib import Path
import pandas as pd

# Define paths
RAW_FILE = Path("data/raw/properties_raw.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PROCESSED_DIR / "properties.csv"

print("=" * 70)
print("CLEANING AND STANDARDIZING PROPERTY DATA")
print("=" * 70)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Source file {RAW_FILE} not found!")

df = pd.read_csv(RAW_FILE)
print(f"Loaded raw records: {len(df)}")

# 1. Select relevant columns
keep_columns = [
    "source_listing_id",
    "city",
    "dealType",
    "category",
    "propertyTitle",
    "price",
    "currency",
    "areaValue",
    "areaUnit",
    "beds",
    "baths",
    "locationText",
    "snippet",
    "url"
]

df_clean = df[keep_columns].copy()

# 2. Handle missing values & types
df_clean["price"] = pd.to_numeric(df_clean["price"], errors="coerce")
df_clean["beds"] = pd.to_numeric(df_clean["beds"], errors="coerce").fillna(0).astype(int)
df_clean["baths"] = pd.to_numeric(df_clean["baths"], errors="coerce").fillna(0).astype(int)
df_clean["areaValue"] = pd.to_numeric(df_clean["areaValue"], errors="coerce")

# Fill string missing values with empty strings
string_cols = ["propertyTitle", "locationText", "snippet", "city", "dealType", "category"]
for col in string_cols:
    df_clean[col] = df_clean[col].fillna("").astype(str)

# 3. Create a combined searchable text field for RAG / Vector Embeddings
df_clean["search_text"] = (
    "City: " + df_clean["city"] + " | " +
    "Type: " + df_clean["dealType"] + " " + df_clean["category"] + " | " +
    "Title: " + df_clean["propertyTitle"] + " | " +
    "Location: " + df_clean["locationText"] + " | " +
    "Beds: " + df_clean["beds"].astype(str) + " | " +
    "Baths: " + df_clean["baths"].astype(str) + " | " +
    "Price: " + df_clean["price"].astype(str) + " " + df_clean["currency"] + " | " +
    "Details: " + df_clean["snippet"]
)

# Drop rows where critical fields (price or city) are missing
df_clean = df_clean.dropna(subset=["price", "city"])

# Save processed dataset
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\nCleaning complete!")
print(f"Cleaned records saved: {len(df_clean)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)
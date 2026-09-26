import os
import logging
import re
import pandas as pd

class PropertyRecommendationEngine:
    def __init__(self, data_path="data/processed/properties.csv"):
        self.data_path = data_path
        self.df = self._load_data()

    def _load_data(self):
        if os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                if "price" in df.columns:
                    df["price"] = pd.to_numeric(df["price"], errors="coerce")
                if "beds" in df.columns:
                    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
                return df
            except Exception as e:
                logging.error(f"PropertyRecommendationEngine failed to load {self.data_path}: {e}")
                return pd.DataFrame()
        logging.error(f"PropertyRecommendationEngine: data file not found at {self.data_path}")
        return pd.DataFrame()

    def recommend(self, city=None, deal_type=None, property_type=None, max_budget=None, min_bedrooms=None, target_area=None, top_n=3):
        if self.df.empty:
            return pd.DataFrame()

        df_filtered = self.df.copy()

        # 1. Filter by City
        if city:
            df_filtered = df_filtered[df_filtered['city'].str.contains(city, case=False, na=False)]

        # 2. Filter by Deal Type
        if deal_type:
            df_filtered = df_filtered[df_filtered['dealType'].str.contains(deal_type, case=False, na=False)]

        # The source dataset groups residential listings under the broad "homes"
        # category, so use listing text to distinguish flats from houses.
        type_terms = {
            "flat": r"\bflat\b|\bflats\b|\bapartment\b|\bapartments\b",
            "house": r"\bhouse\b|\bhouses\b|\bbungalow\b|\bvilla\b",
            "plot": r"\bplot\b|\bplots\b|\bplotting\b",
            "commercial": r"\bcommercial\b|\bshop\b|\boffice\b|\bplaza\b",
        }
        if property_type:
            type_pattern = type_terms.get(str(property_type).strip().lower())
            if type_pattern:
                text_columns = [column for column in ("propertyTitle", "search_text", "snippet") if column in df_filtered.columns]
                if text_columns:
                    listing_text = df_filtered[text_columns].fillna("").astype(str).agg(" ".join, axis=1)
                    df_filtered = df_filtered[listing_text.str.contains(type_pattern, case=False, regex=True, na=False)]

        # 3. Filter by Max Budget
        if max_budget is not None:
            df_filtered = df_filtered[df_filtered['price'] <= float(max_budget)]

        # 4. Filter by Minimum Bedrooms
        if min_bedrooms is not None:
            if 'beds' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['beds'] >= int(min_bedrooms)]

        # 5. Filter by requested property size strictly. Never silently replace
        # a requested 5 Marla/1 Kanal home with a different-sized listing.
        if target_area:
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(kanal|marla|sqft|sq\s*ft)\s*", str(target_area), re.IGNORECASE)
            if match and {"areaValue", "areaUnit"}.issubset(df_filtered.columns):
                amount = float(match.group(1))
                unit = match.group(2).lower().replace(" ", "")
                values = pd.to_numeric(df_filtered["areaValue"], errors="coerce")
                units = df_filtered["areaUnit"].fillna("").astype(str).str.lower().str.replace(" ", "", regex=False)
                if unit in {"kanal", "marla"}:
                    requested_marla = amount * (20 if unit == "kanal" else 1)
                    listing_marla = values.where(units.eq("marla"), values * 20).where(units.isin(["marla", "kanal"]))
                    df_filtered = df_filtered[(listing_marla - requested_marla).abs() <= 0.1]
                else:
                    df_filtered = df_filtered[units.isin(["sqft", "sq.ft", "sqft."]) & ((values - amount).abs() <= 1)]
            else:
                search_column = df_filtered.get("search_text", pd.Series("", index=df_filtered.index)).fillna("").astype(str)
                df_filtered = df_filtered[search_column.str.contains(str(target_area), case=False, na=False)]

        return df_filtered.head(top_n)

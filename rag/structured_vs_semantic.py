from pathlib import Path
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

class Task3StructuredVsSemantic:
    def __init__(self, persist_directory="data/chroma_db"):
        self.csv_path = Path("data/processed/properties.csv")
        self.df = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.property_collection = self.client.get_or_create_collection(
            name="properties_production_rag",
            embedding_function=self.embedding_fn
        )
        self.faq_collection = self.client.get_or_create_collection(
            name="faqs_production_rag",
            embedding_function=self.embedding_fn
        )

    def sql_structured_query(self, city=None, max_price=None, min_beds=None, area_unit=None):
        """
        SQL/Pandas structured filter for exact numerical and categorical constraints.
        """
        print(f"\n[SQL/Pandas Filter] Executing exact filters -> City: {city}, Max Price: {max_price}, Min Beds: {min_beds}")
        if self.df.empty:
            return pd.DataFrame()

        filtered = self.df.copy()
        if city:
            filtered = filtered[filtered["city"].str.lower() == city.lower()]
        if max_price is not None:
            filtered = filtered[filtered["price"] <= max_price]
        if min_beds is not None:
            filtered = filtered[filtered["beds"] >= min_beds]
            
        return filtered

    def vector_semantic_query(self, query: str, collection_type="properties", n_results=2):
        """
        Vector retrieval for descriptions, brochures, and UrduLish FAQs.
        """
        print(f"\n[Vector Semantic Search] Query: '{query}' on collection: {collection_type}")
        col = self.faq_collection if collection_type == "faqs" else self.property_collection
        
        results = col.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"][0], results["metadatas"][0]

if __name__ == "__main__":
    engine = Task3StructuredVsSemantic()
    
    # 1. SQL Structured Test (Exact Price & Beds)
    sql_results = engine.sql_structured_query(city="Lahore", max_price=100000000, min_beds=3)
    print(f"Matched Structured Rows: {len(sql_results)}")
    if not sql_results.empty:
        print(sql_results[["propertyTitle", "price", "beds"]].head(2))

    # 2. Vector Semantic Test (FAQs & Descriptions)
    faq_docs, _ = engine.vector_semantic_query(query="visit ke charges kitne hain?", collection_type="faqs")
    print("\nVector FAQ Match Result:")
    for doc in faq_docs:
        print(f" -> {doc}")
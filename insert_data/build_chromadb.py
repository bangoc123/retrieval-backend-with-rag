import pandas as pd
import ast  # To safely parse string to list
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import argparse
import os 

class DataNotFoundError(Exception):
    def __init__(self):
        super().__init__(f"Please make sure you have valid CSV file in insert_data/data")

def csv_exists(folder_path: str, file_name: str) -> bool:
    """
    Check if a CSV file exists in the given folder.

    Args:
        folder_path (str): Path to the folder (e.g., "./data")
        filename (str): Name of the file (e.g., "products.csv")

    Returns:
        bool: True if file exists, False otherwise
    """
    if not file_name.endswith(".csv"):
        raise ValueError("Filename must end with .csv")
    
    full_path = os.path.join(folder_path, file_name)
    return os.path.isfile(full_path)


def load_csv_to_chromadb(args):
    # Load CSV
    if csv_exists(folder_path="./insert_data/data", file_name=args.csv_path.split('/')[-1]):
        df = pd.read_csv(args.csv_path)
    else:
        raise DataNotFoundError

    if 'combined_infomation' in df.columns:
        df = df.drop(columns=['combined_information'])

    df['combined_information'] = df.apply(lambda row: ', '.join(f"{col}: {row[col]}" for col in df.columns), axis=1)

    # Load sentence embedding model
    model = SentenceTransformer(args.model_name, trust_remote_code=True)

    # Generate embeddings from 'combined_information' column
    df['embedding'] = df['combined_information'].apply(lambda x: model.encode(x).tolist())

    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=args.persist_dir)

    if '/' in args.model_name:
      collection_name = args.model_name.split('/')[1]
    else:
      collection_name = args.model_name
    collection = client.get_or_create_collection(name=collection_name)

    # Add to Chroma
    collection.add(
        ids=df['_id'].astype(str).tolist(),
        documents=df['combined_information'].tolist(),
        embeddings=df['embedding'].tolist(),
        metadatas=[
            {
                "title": row['title'],
                "current_price": row['current_price'],
                "product_promotion": row['product_promotion'],
                "product_specs": row['product_specs'],
                "color_options": row['color_options']
            }
            for _, row in df.iterrows()
        ]
    )

    print(f"{len(df)} items added to collection `{collection_name}`.")

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments to embedding csv data to chromadb vector store")
    parser.add_argument("--csv_path", type=str, required=True, help="Declare CSV data file to embedding.")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="Default directory to store chromadb vector store.")
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-multilingual-base", help="Choose model to embedding.")

    args = parser.parse_args()
    load_csv_to_chromadb(args)
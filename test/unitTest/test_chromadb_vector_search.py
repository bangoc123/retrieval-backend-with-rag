import unittest
import pandas as pd
import os
import sys
import numpy as np
from dotenv import load_dotenv
import shutil
import gdown

# Path setup
current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Debug imports
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

try:
    from rag.core import RAG  # Assumes Reranker has __call__(query, passages) method
    from insert_data import load_csv_to_chromadb
    print("✓ RAG imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise

class ChromaDBVectorSearchTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        # Download data
        url =  "https://drive.google.com/file/d/1LgSkDWQhKx9KYZD-kpmDjgxJRaksVeSs/view?usp=sharing"
        data_dir = os.path.join(project_root, 'data')
        data_path = os.path.join(data_dir, 'data.csv')
        cls.data_path = data_path
        gdown.download(url, data_path, quiet=False, fuzzy=True)

        # Build local chromadb
        load_csv_to_chromadb(csv_path=data_path,persist_dir="./chroma_db")
        llm = None
        cls.rag = RAG(
            type="chromadb",
            embeddingName="Alibaba-NLP/gte-multilingual-base",
            llm=llm,
        )

    def setUp(self):
        self.persist_dir = "./chroma_db"
        self.model_name = "Alibaba-NLP/gte-multilingual-base"

    def test_vector_search(self):
        """Test vector search performance"""
        try:
            results = self.rag.vector_search(
                user_query="Giá iphone 15",
                limit=10
            )
            # Expect result has passage and score
            self.assertIsInstance(results, list, "The results return format must be list")
            self.assertEqual(results[0]['_id'], "666baeb69793e149fe739413")
            self.assertEqual(results[1]['_id'], "666baeb69793e149fe739409")
            self.assertEqual(results[2]['_id'], "666baeb69793e149fe73940a")
            print("Test Passed")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")

    def tearDown(self):
        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        collections = client.list_collections()
        if len(collections) != 0:
            for i in range(len(collections)):
                client.delete_collection(collections[i].name)
        print("Deleted collections after unittest")

        if os.path.exists(self.data_path):
            os.remove(self.data_path)
            print("Removed test CSV")

    @classmethod
    def tearDownClass(cls):
        print("Finished ChromaDBVectorSearchTest\n")
        
if __name__ == "__main__":
    unittest.main()
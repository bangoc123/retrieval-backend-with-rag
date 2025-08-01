import unittest
import pandas as pd
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Path setup
current_path = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_path, '..', '..'))
sys.path.append(project_root)

# Debug imports
print(f"Project root: {project_root}")
print(f"Python path includes: {project_root}")

try:
    from rag.core import RAG  # Assumes Reranker has __call__(query, passages) method
    print("✓ RAG imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    raise

class ChromaDBVectorSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build local chromadb
        llm = None
        cls.rag = RAG(
            type="chromadb",
            embeddingName="Alibaba-NLP/gte-multilingual-base",
            llm=llm,
        )

    def test_vector_search(self):
        """Test vector search performance"""

        results = self.rag.vector_search(
            user_query="Giá iphone 15",
            limit=10
        )
        # Expect result has passage and score

        pass
        
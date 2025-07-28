import pymongo
import google.generativeai as genai
from IPython.display import Markdown
import textwrap
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from typing import Optional, Literal
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import models, QdrantClient

class RAG():
    def __init__(self, 
            llm,
            type: Literal['mongodb', 'qdrant'],
            mongodbUri: Optional[str] = None,
            qdrant_api: Optional[str] = None,
            qdrant_url: Optional[str] = None,
            dbName: Optional[str] = None,
            dbCollection: Optional[str] = None,
            embeddingName: str ='keepitreal/vietnamese-sbert',
        ):
        self.type = type
        if self.type == 'mongodb':
            self.client = pymongo.MongoClient(mongodbUri)
            self.db = self.client[dbName] 
            self.collection = self.db[dbCollection]
        elif self.type == 'qdrant':
            self.qdrant_api = qdrant_api
            self.qdrant_url = qdrant_url
            self.qdrant_collection = embeddingName.split('/')[1]
            self.client = QdrantClient(
                            url=self.qdrant_url,
                            api_key=self.qdrant_api
                            ) 
        self.embedding_model = SentenceTransformerEmbedding(
            EmbeddingConfig(name=embeddingName)
        )
        self.llm = llm

    def get_embedding(self, text):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def _collection_exists(self):           
        """
        Check if Qdrant collection is exists
        """

        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            return self.collection_name in collection_names
        except Exception as e:
            return False
    def vector_search(
            self, 
            user_query: str, 
            limit=4):
        """
        Perform a vector search in the MongoDB collection or Qdrant collection based on the user query.

        Args:
        user_query (str): The user's query string.

        Returns:
        list: A list of matching documents.
        """

        # Generate embedding for the user query
        query_embedding = self.get_embedding(user_query)

        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        if self.type == 'qdrant':
            if self._collection_exists:
                hits = self.client.search(
                    collection_name=self.qdrant_collection,
                    query_vector=self.embedding_model.encode(user_query).tolist(),
                    limit=limit
                )               
                results = []
                for hit in hits:
                    results.append(hit.payload)
                return results
            else: 
                print(f"Collection is not exist")
                return
        elif self.type == 'mongodb':
            vector_search_stage = {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": 400,
                    "limit": limit,
                }
            }

            unset_stage = {
                "$unset": "embedding" 
            }

            project_stage = {
                "$project": {
                    "_id": 1,  
                    "title": 1, 
                    # "product_specs": 1,
                    "color_options": 1,
                    "current_price": 1,
                    "product_promotion": 1,
                    "score": {
                        "$meta": "vectorSearchScore"
                    }
                }
            }

            pipeline = [vector_search_stage, unset_stage, project_stage]

            # Execute the search
            results = self.collection.aggregate(pipeline)
    
            return list(results)

    def enhance_prompt(self, query):
        get_knowledge = self.vector_search(query, 10)
        enhanced_prompt = ""
        i = 0
        for result in get_knowledge:
            if result.get('title'):
                i += 1
                enhanced_prompt += f"\n{i}) Tên: {result.get('title')}"

                # Price 
                if result.get('current_price'):
                    enhanced_prompt += f", Giá: {result.get('current_price')}"
                else:
                    enhanced_prompt += f", Giá: Không có thông "
                # Promotion
                if result.get('product_promotion'):
                    enhanced_prompt += f", Ưu đãi: {result['product_promotion']}"
                else:
                    enhanced_prompt += ", Ưu đãi: Không có thông tin"

                # Specifications
                if result.get('product_specs'):
                    enhanced_prompt += f", Thông số: {result['product_specs']}"
                else:
                    enhanced_prompt += ", Thông số: Không có thông tin"

                # Color options
                if result.get('color_options'):
                    enhanced_prompt += f", Màu sắc: {result['color_options']}"
                else:
                    enhanced_prompt += ", Màu sắc: Không có thông tin"
        return enhanced_prompt

    def generate_content(self, prompt):
        return self.llm.generate_content(prompt)

    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

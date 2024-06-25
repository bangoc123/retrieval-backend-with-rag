import pymongo
import google.generativeai as genai
from IPython.display import Markdown
import textwrap
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig

class RAG():
    def __init__(self, 
            mongodbUri: str,
            dbName: str,
            dbCollection: str,
            llmApiKey: str,
            llmName: str ='gemini-1.5-pro',
            embeddingName: str ='keepitreal/vietnamese-sbert',
        ):
        self.client = pymongo.MongoClient(mongodbUri)
        self.db = self.client[dbName] 
        self.collection = self.db[dbCollection]
        self.embedding_model = SentenceTransformerEmbedding(
            EmbeddingConfig(name=embeddingName)
        )

        # config llm
        genai.configure(api_key=llmApiKey)
        self.llm = genai.GenerativeModel(llmName)

    def get_embedding(self, text):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def vector_search(
            self, 
            user_query: str, 
            limit=4):
        """
        Perform a vector search in the MongoDB collection based on the user query.

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
                "_id": 0,  
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
            if result.get('current_price'):
                i += 1
                enhanced_prompt += f"\n {i}) Tên: {result.get('title')}"
                
                if result.get('current_price'):
                    enhanced_prompt += f", Giá: {result.get('current_price')}"
                else:
                    # Mock up data
                    # Retrieval model pricing from the internet.
                    enhanced_prompt += f", Giá: Liên hệ để trao đổi thêm!"
                
                if result.get('product_promotion'):
                    enhanced_prompt += f", Ưu đãi: {result.get('product_promotion')}"
        return enhanced_prompt

    def generate_content(self, prompt):
        return self.llm.generate_content(prompt)

    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

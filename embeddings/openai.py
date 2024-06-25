import os
from typing import Optional, Union, List
from pydantic.v1 import BaseModel, Field, PrivateAttr
from embeddings import APIBaseEmbedding
import openai
from dotenv import load_dotenv
load_dotenv()

class OpenAIEmbedding(APIBaseEmbedding):
    def __init__(
            self,
            name: str = "text-embedding-ada-002",
            dimensions: int = 768,
            token_limit: int = 8192,
            baseUrl: str = None,
            apiKey: str = None,
            orgId: str = None,
        ):
        super().__init__(name=name, baseUrl=baseUrl, apiKey=apiKey)
        self.dimensions = dimensions
        self.apiKey = apiKey or os.getenv("OPENAI_API_KEY")
        self.orgId = orgId or os.getenv("OPENAI_ORG_ID")
        self.baseUrl = orgId or os.getenv("OPENAI_BASE_URL")
        
        if not self.apiKey:
            raise ValueError("The OpenAI API key must not be 'None'.")
        
        try:
            self.client = openai.Client(
                base_url=self.baseUrl, api_key=self.apiKey, organization=self.orgId 
            )
        except Exception as e:
            raise ValueError(
                f"OpenAI API client failed to initialize. Error: {e}"
            ) from e

    def encode(self, docs: List[str]):
        try:
            embeds = self.client.embeddings.create(
                    input=docs,
                    model=self.name,
                    dimensions=self.dimensions,
                )
            embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
            return embeddings
        except Exception as e:
            raise ValueError(
                f"Failed to get embeddings. Error details: {e}"
            ) from e
import os
from typing import Optional, Union, List
from mistralai.client import MistralClient
from embeddings import APIBaseEmbedding

class MistralEmbedding(APIBaseEmbedding):
    def __init__(
            self,
            name: str = "mistral-embed",
            apiKey: str = None,
        ):
        super().__init__(name=name, apiKey=apiKey)
        self.apiKey = apiKey or os.getenv("MISTRAL_KEY")
        
        if not self.apiKey:
            raise ValueError("The Mistral API key must not be 'None'.")
        
        try:
            self.client = MistralClient(
                api_key=self.apiKey
            )
        except Exception as e:
            raise ValueError(
                f"Mistral API client failed to initialize. Error: {e}"
            ) from e

    def encode(self, docs: List[str]):
        try:
            embeds = self.client.embeddings(
                    input=docs,
                    model=self.name,
                )
            embeddings = [embeds_obj.embedding for embeds_obj in embeds.data]
            return embeddings
        except Exception as e:
            raise ValueError(
                f"Failed to get embeddings. Error details: {e}"
            ) from e
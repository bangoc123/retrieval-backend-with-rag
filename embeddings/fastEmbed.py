import os
from typing import Optional, Union, List
from fastembed import TextEmbedding
from embeddings import BaseEmbedding

class FastEmbedding(BaseEmbedding):
    def __init__(
            self,
            # Multilingual model
            name: str = 'BAAI/bge-m3',
            max_length:int = 512
        ):
        super().__init__(name=name)
        
        try:
            self.embedding_model = TextEmbedding(
                name=name,
                max_length=max_length
            )
        except Exception as e:
            raise ValueError(
                f"Fastembed failed to initialize. Error: {e}"
            ) from e

    def encode(self, docs: List[str]):
        try:
            embeds = self.embedding_model.embed(docs)
            embeddings: List[List[float]] = [e.tolist() for e in embeds]
            return embeddings
        except Exception as e:
            raise ValueError(
                f"Failed to get embeddings. Error details: {e}"
            ) from e
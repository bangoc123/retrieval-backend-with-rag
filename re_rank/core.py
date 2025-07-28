from sentence_transformers import CrossEncoder
import numpy as np

class Reranker():
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = CrossEncoder(model_name)

    def __call__(self, query: str, passages: list[str]) -> list[str]:
     
        # Return just the passages in ranked order
        return []
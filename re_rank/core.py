
from sentence_transformers import CrossEncoder
import numpy as np

class Reranker():
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = CrossEncoder(model_name)

    def __call__(self, query: str, passages: list[str]) -> tuple[list[float], list[str]]:
        # Combine query and passages into pairs
        query_passage_pairs = [[query, passage] for passage in passages]

        # Get scores from the reranker model
        scores = self.reranker.predict(query_passage_pairs)

        # Sort passages based on scores
        ranked_passages = [passage for _, passage in sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)]
        ranked_scores = sorted(scores, reverse=True)
        
        # Convert scores to standard Python floats
        ranked_scores = [float(score) for score in ranked_scores]
        # Return just the passages in ranked order
        return ranked_scores, ranked_passages
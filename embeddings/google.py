##https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings


from embeddings import APIBaseEmbedding
import os
import openai
from typing import Any, List, Optional
from dotenv import load_dotenv
load_dotenv()

class GoogleEmbedding(APIBaseEmbedding):
    def __init__(
        self,
        name: str = "textembedding-gecko@003",
        dimensions: int = 768,
        token_limit: int = 8192,
        baseUrl: str = None,
        apiKey: str = None,
        projectId: str = None,
        location: str = None,
    ):
        super().__init__(name=name, baseUrl=baseUrl, apiKey=apiKey)
        self.name = name

        try:
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel
        except ImportError:
            raise ImportError(
                "To use GoogleEmbedding, please install the Google Cloud and Vertex AI libraries "
                "You can do this with the following command: "
                "`pip install google-cloud-aiplatform vertexai-language-models`"
            )
        
        projectId = projectId or os.getenv("GOOGLE_PROJECT_ID")
        location = location or os.getenv("GOOGLE_LOCATION", "us-central1")
        baseUrl = baseUrl or os.getenv("GOOGLE_BASE_URL")

        if projectId is None:
            raise ValueError("Google Project ID cannot be null.")
        
        try:
            aiplatform.init(
                project=projectId, location=location, api_endpoint=baseUrl
            )
            self.client = TextEmbeddingModel.from_pretrained(self.name)
        except Exception as err:
            raise ValueError(
                f"Failed to initialize Google AI Platform client. Error: {err}"
            ) from err

    def encode(self, docs: List[str]):
        try:
            embeddings = self.client.get_embeddings(docs)
            return [embedding.values for embedding in embeddings]
        except Exception as e:
            raise ValueError(f"Google AI Platform API call failed. Error: {e}") from e
        
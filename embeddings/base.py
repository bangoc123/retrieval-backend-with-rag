from pydantic.v1 import BaseModel, Field, validator
from typing import Any, Optional

class EmbeddingConfig(BaseModel):
    name: str = Field(..., description="The name of the SentenceTransformer model")

    @validator('name')
    def check_model_name(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Model name must be a non-empty string")
        return value

class BaseEmbedding():
    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def encode(self, text: str):
        raise NotImplementedError("The encode method must be implemented by subclasses")


class APIBaseEmbedding(BaseEmbedding):
    baseUrl: str
    apiKey: str

    def __init__(self, name: str = None, baseUrl: str = None, apiKey: str = None):
        super().__init__(name)
        self.baseUrl = baseUrl
        self.apiKey = apiKey

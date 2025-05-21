from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, utterances: List[str]) -> List[List[float]]:
        return self.model.encode(utterances, convert_to_numpy=True)
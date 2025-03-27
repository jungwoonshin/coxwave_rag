
from typing import List

import numpy as np



class EmbeddingGenerator:
    def __init__(self, model_name: str = "openai/text-embedding-3-small", api_key: str = None):
        """Initialize the embedding generator with model name and API key."""
        pass
        
    def generate(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass
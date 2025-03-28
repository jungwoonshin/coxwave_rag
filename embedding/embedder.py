import torch
from typing import List, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Class for creating embeddings from text using HuggingFace models.
    """
    def __init__(self, model_name: str, token: str):
        """
        Initialize the TextEmbedder with a specific embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            token: HuggingFace API token
        """
        self.model_name = model_name
        self.token = token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            self.model = AutoModel.from_pretrained(model_name, token=token).to(self.device)
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on model outputs, taking into account the attention mask.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from the tokenization
            
        Returns:
            Pooled embeddings
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for a text or list of texts.
        
        Args:
            texts: A single text string or a list of text strings
            
        Returns:
            NumPy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize and prepare for the model
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Perform pooling and normalize
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
"""
Text embedding module supporting both single and batch embedding operations.
"""
import logging
import os
import time
from typing import List, Optional
import openai

logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Class for generating text embeddings using OpenAI models.
    Supports both single and batch embedding operations.
    """
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: int = 1
    ):
        """
        Initialize the TextEmbedder.
        
        Args:
            model_name: Name of the embedding model to use
            api_key: OpenAI API key (will use environment variable if not provided)
            retry_count: Number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        # Set API key
        if api_key:
            openai.api_key = api_key
        elif "OPENAI_API_KEY" in os.environ:
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            logger.warning("No OpenAI API key provided. Please set OPENAI_API_KEY environment variable.")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of embedding values
        """
        for attempt in range(self.retry_count):
            try:
                # Make OpenAI API call
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=text
                )
                
                # Extract embedding from response
                embedding = response.data[0].embedding
                return embedding
                
            except Exception as e:
                logger.warning(f"Error generating embedding (attempt {attempt+1}/{self.retry_count}): {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Failed to generate embedding after multiple attempts")
                    raise
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a batch of texts using efficient batching.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors in the same order as input texts
        """
        if not texts:
            return []
        
        # Process in optimal batch sizes
        if len(texts) > self.max_batch_size:
            logger.info(f"Input size {len(texts)} exceeds max batch size {self.max_batch_size}. Processing in multiple batches.")
            all_embeddings = []
            
            # Process in chunks of max_batch_size
            for i in range(0, len(texts), self.max_batch_size):
                batch = texts[i:i + self.max_batch_size]
                batch_embeddings = self._process_single_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
            return all_embeddings
        else:
            # Process as a single batch if within size limit
            return self._process_single_batch(texts)
    
    def _process_single_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Process a single batch of texts that fits within the API limits.
        
        Args:
            texts: List of texts to generate embeddings for (should be <= max_batch_size)
            
        Returns:
            List of embedding vectors
        """
        for attempt in range(self.retry_count):
            try:
                start_time = time.time()
                
                # Make OpenAI API call with batch input
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=texts,
                    encoding_format="float"  # Explicitly request float format
                )
                
                # Log performance metrics
                duration = time.time() - start_time
                logger.debug(f"Generated {len(texts)} embeddings in {duration:.2f}s ({len(texts)/duration:.1f} texts/sec)")
                
                # Extract embeddings from response, ensuring proper order
                embeddings = [data.embedding for data in sorted(response.data, key=lambda x: x.index)]
                
                # Validate response
                if len(embeddings) != len(texts):
                    logger.warning(f"Received {len(embeddings)} embeddings but expected {len(texts)}")
                    
                return embeddings
                
            except Exception as e:
                logger.warning(f"Error generating batch embeddings (attempt {attempt+1}/{self.retry_count}): {str(e)}")
                
                # Handle rate limiting
                if "rate limit" in str(e).lower():
                    # Exponential backoff
                    backoff_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate limited. Backing off for {backoff_time}s before retry.")
                    time.sleep(backoff_time)
                    continue
                
                # Handle token limit or other capacity issues
                if "too many" in str(e).lower() and len(texts) > 1:
                    logger.info(f"Batch size {len(texts)} too large. Splitting and processing in smaller batches.")
                    
                    # Split the batch and process recursively
                    mid = len(texts) // 2
                    first_half = self._process_single_batch(texts[:mid])
                    second_half = self._process_single_batch(texts[mid:])
                    return first_half + second_half
                
                # General retry with delay
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("Failed to generate batch embeddings after multiple attempts")
                    raise
    
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of embeddings for the current model.
        
        Returns:
            Number of dimensions
        """
        # Known dimensions for common models
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if self.model_name in model_dimensions:
            return model_dimensions[self.model_name]
        
        # For unknown models, generate a sample embedding and check its length
        sample_text = "This is a sample text to determine embedding dimensions."
        sample_embedding = self.get_embedding(sample_text)
        return len(sample_embedding)
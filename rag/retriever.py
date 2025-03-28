from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility
from typing import List, Tuple, Dict, Any
import numpy as np
import logging
import time
import uuid

from embedding.embedder import TextEmbedder

logger = logging.getLogger(__name__)

class MilvusRetriever:
    """
    Class for setting up and querying a Milvus vector database for RAG.
    """
    def __init__(
        self,
        embedder: TextEmbedder,
        host: str,
        port: str,
        collection_name: str,
        vector_dim: int
    ):
        """
        Initialize the Milvus retriever with connection parameters.
        
        Args:
            embedder: TextEmbedder instance for creating embeddings
            host: Milvus host address
            port: Milvus port
            collection_name: Name of the collection to use
            vector_dim: Dimension of the embedding vectors
        """
        self.embedder = embedder
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.collection = None
        
        # Connect to Milvus
        try:
            connections.connect(host=host, port=port)
            logger.info(f"Connected to Milvus at {host}:{port}")
        except Exception as e:
            logger.error(f"Error connecting to Milvus: {e}")
            raise
    
    def _create_collection(self) -> Collection:
        """
        Create a new collection in Milvus for storing FAQ embeddings.
        
        Returns:
            Milvus Collection object
        """
        # Define fields for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        # Create collection schema
        schema = CollectionSchema(fields=fields, description="FAQ Collection")
        
        # Create collection
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create IVF_FLAT index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection {self.collection_name} with vector dimension {self.vector_dim}")
        
        return collection
    
    def _get_or_create_collection(self) -> Collection:
        """
        Get the existing collection or create a new one if it doesn't exist.
        
        Returns:
            Milvus Collection object
        """
        if utility.has_collection(self.collection_name):
            collection = Collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        else:
            collection = self._create_collection()
        
        return collection
    
    def setup_collection(self, qa_pairs: List[Tuple[str, str]]):
        """
        Set up the Milvus collection with FAQ data.
        
        Args:
            qa_pairs: List of (question, answer) tuples
        """
        self.collection = self._get_or_create_collection()
        
        # Check if collection is empty
        if self.collection.num_entities > 0:
            logger.info(f"Collection already has {self.collection.num_entities} entities. Skipping insertion.")
            self.collection.load()
            return
        
        # Prepare data for insertion
        questions = [q for q, _ in qa_pairs]
        answers = [a for _, a in qa_pairs]
        
        # Generate embeddings for questions
        embeddings = self.embedder.get_embeddings(questions)
        
        # Generate unique IDs
        ids = [str(uuid.uuid4()) for _ in range(len(qa_pairs))]
        
        # Insert data in batches
        data = [ids, questions, answers, embeddings]
        
        try:
            self.collection.insert(data)
            logger.info(f"Inserted {len(qa_pairs)} FAQ pairs into Milvus collection")
            
            # Load collection into memory for searching
            self.collection.load()
        except Exception as e:
            logger.error(f"Error inserting data into Milvus: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve the most similar FAQ pairs for a given query.
        
        Args:
            query: User query string
            top_k: Number of most similar results to return
            
        Returns:
            List of dictionaries containing question, answer, and similarity score
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call setup_collection first.")
        
        # Generate embedding for the query
        query_embedding = self.embedder.get_embeddings(query)
        
        # Search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Perform the search
        results = self.collection.search(
            data=[query_embedding[0].tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["question", "answer"]
        )
        
        # Process search results
        retrieved_results = []
        for hits in results:
            for hit in hits:
                retrieved_results.append({
                    "question": hit.entity.get("question"),
                    "answer": hit.entity.get("answer"),
                    "score": float(hit.score)  # Convert to float for JSON serialization
                })
        
        return retrieved_results
import logging
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class ChromaRetriever:
    """
    Retriever class using ChromaDB for vector search with embedding caching.
    Compatible with the cache created by initialize_db.py.
    """
    def __init__(
        self,
        embedder=None,
        collection_name: str = "document_store",
        db_path: str = "chroma_data",  # Data directory for persistent ChromaDB
        cache_dir: str = "embedding_cache",  # Directory for embedding cache
        embedding_dimension: int = 1536,  # For text-embedding-3-small
        top_k: int = 10,
        metric_type: str = "cosine"
    ):
        """
        Initialize the ChromaRetriever.
        
        Args:
            embedder: Text embedder instance (optional)
            collection_name: Name of the collection to use
            db_path: Path to the ChromaDB data directory
            cache_dir: Path to the embedding cache directory
            embedding_dimension: Dimension of the embeddings
            top_k: Default number of results to return
            metric_type: Similarity metric to use (cosine, euclidean, dot_product)
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_dimension = embedding_dimension
        self.metric_type = metric_type
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        try:
            # Make sure the persistence directory exists
            os.makedirs(db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            logger.info(f"Connecting to ChromaDB at {db_path}")
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            
            # Initialize cache
            self._init_cache()
            
            # Get or create collection
            self._ensure_collection_exists()
            
        except Exception as e:
            logger.error(f"Error initializing ChromaRetriever: {e}")
            raise
    
    def _init_cache(self):
        """Initialize the embedding cache system"""
        # Create metadata file if it doesn't exist
        metadata_path = self.cache_dir / "metadata.json"
        self.cache_metadata = {}
        self.model_name = "unknown"
        
        if metadata_path.exists():
            try:
                # Load existing metadata
                with open(metadata_path, "r") as f:
                    self.cache_metadata = json.load(f)
                self.model_name = self.cache_metadata.get("model", "unknown")
                logger.info(f"Using cache for model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")
                
            # Check if model changed
            if self.embedder and self.model_name != self.embedder.model_name:
                logger.warning(f"Embedding model mismatch: cache uses {self.model_name}, current is {self.embedder.model_name}")
                self.model_name = self.embedder.model_name
        else:
            # Create new metadata
            if self.embedder:
                self.model_name = self.embedder.model_name
            with open(metadata_path, "w") as f:
                json.dump({"model": self.model_name}, f)
            logger.info(f"Created new cache metadata for model: {self.model_name}")
        
        # Load model-specific cache
        self.embedding_cache = {}
        self.cache_file = self.cache_dir / f"{self.model_name}_embeddings.json"
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Error loading embeddings cache: {e}")
                self.embedding_cache = {}
        else:
            logger.info(f"No existing cache file found at {self.cache_file}, will create as needed")
    
    def _get_embedding_with_cache(self, text: str) -> List[float]:
        """
        Get embedding for text with caching.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding vector
        """
        if not self.embedder:
            raise ValueError("No embedder provided for generating embeddings")
        
        # Hash the text to use as cache key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Check if we have this embedding cached
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Generate new embedding
        vector = self.embedder.get_embedding(text)
        
        # Update cache
        self.embedding_cache[text_hash] = vector
        
        # Save cache periodically (every 10 new embeddings)
        if len(self.embedding_cache) % 10 == 0:
            try:
                with open(self.cache_file, "w") as f:
                    json.dump(self.embedding_cache, f)
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
        
        return vector
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists or create it
            try:
                # Try to get the collection first
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except:
                # Create collection if it doesn't exist
                logger.info(f"Creating collection: {self.collection_name}")
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": self.metric_type}  # Set distance metric
                )
                
        except Exception as e:
            logger.error(f"Error creating/getting collection: {e}")
            raise
    
    def setup_collection(self, qa_pairs: List[Dict[str, Any]]) -> None:
        """
        Set up the collection with FAQ data.
        
        Args:
            qa_pairs: List of question-answer pairs with or without embeddings
        """
        try:
            # Delete collection if it exists
            try:
                self.chroma_client.delete_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            except:
                logger.info(f"No existing collection to drop")
            
            # Recreate the collection
            self._ensure_collection_exists()
            
            # Prepare data for insertion
            ids = []
            questions = []
            answers = []
            embeddings = []
            
            for i, pair in enumerate(qa_pairs):
                # Get or generate embedding
                if "vector" in pair and pair["vector"]:
                    # Use provided vector (from initialize_db.py)
                    vector = pair["vector"]
                    
                    # Also update the cache with this vector
                    question = pair.get("question", "")
                    text_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
                    self.embedding_cache[text_hash] = vector
                elif "question" in pair:
                    # Use cached embedding generation
                    vector = self._get_embedding_with_cache(pair["question"])
                else:
                    logger.warning(f"Skipping pair {i}: No vector or question")
                    continue
                
                # Get ID or use index
                doc_id = str(pair.get("id", i))  # ChromaDB requires string IDs
                
                # Add to batch
                ids.append(doc_id)
                questions.append(pair.get("question", ""))
                answers.append(pair.get("answer", ""))
                embeddings.append(vector)
                
                # Log progress periodically
                if (i + 1) % 100 == 0:
                    logger.info(f"Prepared {i + 1} documents for insertion")
            
            # Save updated cache
            with open(self.cache_file, "w") as f:
                json.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
            
            # Insert in batches (ChromaDB has batch size limits)
            BATCH_SIZE = 500
            for i in range(0, len(ids), BATCH_SIZE):
                batch_ids = ids[i:i+BATCH_SIZE]
                batch_questions = questions[i:i+BATCH_SIZE]
                batch_answers = answers[i:i+BATCH_SIZE]
                batch_embeddings = embeddings[i:i+BATCH_SIZE]
                
                # Build metadata list
                metadata_batch = [{"answer": answer} for answer in batch_answers]
                
                # Add embeddings to collection
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_questions,
                    metadatas=metadata_batch
                )
                
                logger.info(f"Inserted batch of {len(batch_ids)} documents")
                
            logger.info(f"Inserted total of {len(ids)} documents into collection {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def retrieve(self, query: str = None, query_vector: List[float] = None, top_k: Optional[int] = None) -> List[Dict]:
        """
        Retrieve similar documents using vector search.
        
        Args:
            query: The query text (used if query_vector not provided)
            query_vector: The query embedding vector
            top_k: Number of results to return (defaults to self.top_k)
            
        Returns:
            List of similar documents with metadata and scores
        """
        if top_k is None:
            top_k = self.top_k
            
        # Get query vector from text if needed
        if query_vector is None:
            if query is None:
                raise ValueError("Either query_vector or query must be provided")
            
            # Use cached embedding for query
            query_vector = self._get_embedding_with_cache(query)
            
        try:
            # Execute search
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    # For cosine, convert distance to similarity (1 - distance)
                    if self.metric_type == "cosine":
                        score = 1 - results["distances"][0][i]
                    else:
                        # For other metrics, use the negative of distance
                        score = -results["distances"][0][i]
                    
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "question": results["documents"][0][i],
                        "answer": results["metadatas"][0][i]["answer"],
                        "score": score
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns:
            Document count
        """
        try:
            # Get collection count
            return self.collection.count()
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
            
    def batch_process_questions(self, questions: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Process a batch of questions to get their embeddings with caching.
        
        Args:
            questions: List of questions to process
            batch_size: Size of batches to process
            
        Returns:
            List of embedding vectors
        """
        if not self.embedder:
            raise ValueError("No embedder provided for generating embeddings")
        
        # Identify questions that need embeddings
        questions_to_embed = []
        question_hashes = []
        result_map = {}
        
        for i, question in enumerate(questions):
            question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            
            if question_hash in self.embedding_cache:
                # Use cached embedding
                result_map[i] = self.embedding_cache[question_hash]
            else:
                # Need to generate embedding
                questions_to_embed.append(question)
                question_hashes.append(question_hash)
                result_map[i] = None
        
        # Process questions that need embeddings in batches
        if questions_to_embed:
            logger.info(f"Generating embeddings for {len(questions_to_embed)} questions")
            
            for i in range(0, len(questions_to_embed), batch_size):
                batch = questions_to_embed[i:i+batch_size]
                batch_hashes = question_hashes[i:i+batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = self.embedder.get_embeddings_batch(batch)
                
                # Update cache
                for j, embedding in enumerate(batch_embeddings):
                    hash_key = batch_hashes[j]
                    self.embedding_cache[hash_key] = embedding
                
                logger.info(f"Processed batch of {len(batch)} questions")
            
            # Save updated cache
            with open(self.cache_file, "w") as f:
                json.dump(self.embedding_cache, f)
            logger.info(f"Updated embedding cache with {len(self.embedding_cache)} entries")
            
            # Fill in the missing embeddings in the result_map
            for i, question in enumerate(questions):
                if result_map[i] is None:
                    question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
                    result_map[i] = self.embedding_cache[question_hash]
        
        # Assemble results in original order
        results = [result_map[i] for i in range(len(questions))]
        return results
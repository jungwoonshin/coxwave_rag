#!/usr/bin/env python3
"""
Script to initialize the Chroma database with FAQ data.
This script uses batch processing for embedding generation and creates a cache
that can be reused by ChromaRetriever.
"""
import logging
import os
import json
import hashlib
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Import necessary components
from config.setting import (
    EMBEDDING_MODEL_NAME, OPENAI_API_KEY, DATA_PATH
)
from data.loader import DataLoader
from embedding.embedder import TextEmbedder
from rag.retriever import ChromaRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the ChromaDB database with FAQ data using batched embedding generation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize ChromaDB with FAQ data")
    parser.add_argument("--batch-size", type=int, default=2048, 
                        help="Batch size for embedding generation (default: 2048)")
    parser.add_argument("--force-refresh", action="store_true", 
                        help="Force regeneration of all embeddings")
    parser.add_argument("--collection-name", type=str, default="document_store", 
                        help="Name of the collection (default: document_store)")
    parser.add_argument("--db-path", type=str, default="./chroma_data", 
                        help="Path to the ChromaDB directory (default: ./chroma_data)")
    parser.add_argument("--cache-dir", type=str, default="./embedding_cache", 
                        help="Path to the embedding cache directory (default: ./embedding_cache)")
    args = parser.parse_args()
    
    start_time = time.time()
    logger.info("Starting ChromaDB database initialization...")
    
    # Initialize data loader
    data_loader = DataLoader(DATA_PATH)
    logger.info(f"Loading data from {DATA_PATH}")
    
    # Load FAQ data
    qa_pairs = data_loader.get_question_answer_pairs()
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from data source")
    
    # Initialize text embedder
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME, OPENAI_API_KEY)
    logger.info(f"Initialized text embedder with model: {EMBEDDING_MODEL_NAME}")
    
    # Create cache directory if it doesn't exist
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)
    
    # Create model-specific cache file name
    model_cache_file = cache_dir / f"{EMBEDDING_MODEL_NAME}_embeddings.json"
    
    # Create a cache metadata file to track the model used
    cache_metadata_path = cache_dir / "metadata.json"
    if not cache_metadata_path.exists() or args.force_refresh:
        with open(cache_metadata_path, "w") as f:
            json.dump({"model": EMBEDDING_MODEL_NAME}, f)
        logger.info(f"Created/updated cache metadata file")
    
    # Load existing cache if available
    cache_data = {}
    if model_cache_file.exists() and not args.force_refresh:
        try:
            with open(model_cache_file, "r") as f:
                cache_data = json.load(f)
            logger.info(f"Loaded {len(cache_data)} cached embeddings from {model_cache_file}")
        except Exception as e:
            logger.warning(f"Error loading cache file: {e}")
            cache_data = {}
    
    # Identify questions that need embeddings
    questions_to_embed = []
    question_hashes = []
    
    for i, pair in enumerate(qa_pairs):
        question = pair.get("question", "")
        question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        if question_hash not in cache_data or args.force_refresh:
            questions_to_embed.append(question)
            question_hashes.append(question_hash)
    
    logger.info(f"Found {len(questions_to_embed)} questions that need new embeddings")
    
    # Process embeddings in batches
    if questions_to_embed:
        total_batches = (len(questions_to_embed) + args.batch_size - 1) // args.batch_size
        
        for batch_idx in range(0, len(questions_to_embed), args.batch_size):
            batch_num = batch_idx // args.batch_size + 1
            batch = questions_to_embed[batch_idx:batch_idx + args.batch_size]
            batch_hashes = question_hashes[batch_idx:batch_idx + args.batch_size]
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} questions)")
            
            # Generate embeddings for the batch
            start_batch_time = time.time()
            batch_embeddings = embedder.get_embeddings_batch(batch)
            batch_time = time.time() - start_batch_time
            
            # Log performance
            questions_per_second = len(batch) / batch_time if batch_time > 0 else 0
            logger.info(f"Processed {len(batch)} embeddings in {batch_time:.2f}s ({questions_per_second:.2f} q/s)")
            
            # Add to cache
            for i, embedding in enumerate(batch_embeddings):
                if i < len(batch_hashes):
                    cache_data[batch_hashes[i]] = embedding
            
            # Save cache periodically to avoid losing work
            if batch_num % 5 == 0 or batch_num == total_batches:
                with open(model_cache_file, "w") as f:
                    json.dump(cache_data, f)
                logger.info(f"Saved cache with {len(cache_data)} embeddings")
        
        # Final save of cache
        with open(model_cache_file, "w") as f:
            json.dump(cache_data, f)
        logger.info(f"Completed and saved {len(cache_data)} embeddings to cache at {model_cache_file}")
    else:
        logger.info("All embeddings found in cache, no new embeddings needed")
    
    # Initialize ChromaDB retriever with the same cache directory
    db_path = os.path.abspath(args.db_path)
    logger.info(f"Initializing ChromaDB at: {db_path}")
    
    retriever = ChromaRetriever(
        embedder=embedder,
        collection_name=args.collection_name,
        db_path=db_path,
        cache_dir=args.cache_dir,  # Pass the same cache directory
        embedding_dimension=embedder.get_dimensions()
    )
    
    # Prepare documents with vectors from cache
    processed_pairs = []
    for i, pair in enumerate(qa_pairs):
        question = pair.get("question", "")
        question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
        
        if question_hash in cache_data:
            # Get embedding from cache
            vector = cache_data[question_hash]
            
            # Create document with embedding
            processed_pair = {
                "id": i,
                "question": question,
                "answer": pair.get("answer", ""),
                "vector": vector
            }
            processed_pairs.append(processed_pair)
        else:
            logger.warning(f"Missing embedding for question: {question[:50]}...")
    
    # Set up the collection with processed QA pairs
    retriever.setup_collection(processed_pairs)
    
    # Get document count to verify
    doc_count = retriever.get_document_count()
    
    # Report timing
    elapsed_time = time.time() - start_time
    logger.info(f"ChromaDB initialized with {doc_count} documents in {elapsed_time:.2f} seconds")
    logger.info("Database initialization completed successfully")

if __name__ == "__main__":
    main()
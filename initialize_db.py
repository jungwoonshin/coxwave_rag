"""
Script to initialize the Milvus database with FAQ data.
This is useful for pre-loading the database before starting the API server.
"""
import logging
from config.setting import (
    HF_TOKEN, EMBEDDING_MODEL_NAME, 
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, VECTOR_DIM,
    DATA_PATH
)
from data.loader import DataLoader
from embedding.embedder import TextEmbedder
from rag.retriever import MilvusRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the Milvus database with FAQ data."""
    logger.info("Starting database initialization...")
    
    # Initialize data loader
    data_loader = DataLoader(DATA_PATH)
    
    # Initialize text embedder
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME, HF_TOKEN)
    
    # Initialize Milvus retriever
    retriever = MilvusRetriever(
        embedder=embedder,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=MILVUS_COLLECTION,
        vector_dim=VECTOR_DIM
    )
    
    # Load FAQ data and set up the collection
    qa_pairs = data_loader.get_question_answer_pairs()
    retriever.setup_collection(qa_pairs)
    
    logger.info("Database initialization completed successfully")

if __name__ == "__main__":
    main()


import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.setting import (
    HF_TOKEN, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME, 
    MILVUS_HOST, MILVUS_PORT, MILVUS_COLLECTION, VECTOR_DIM,
    DATA_PATH, API_HOST, API_PORT
)
from data.loader import DataLoader
from embedding.embedder import TextEmbedder
from rag.retriever import MilvusRetriever
from llm.model import LlamaModel
from api.router import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize global objects
data_loader = None
embedder = None
retriever = None
llm_model = None

def initialize_components():
    """Initialize all components of the FAQ system."""
    global data_loader, embedder, retriever, llm_model
    
    logger.info("Initializing components...")
    
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
    
    # Initialize LLM model
    llm_model = LlamaModel(LLM_MODEL_NAME, HF_TOKEN)
    
    logger.info("All components initialized successfully")

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="FAQ Answering System",
        description="A RAG-based FAQ answering system using Llama and Milvus",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api")
    
    @app.on_event("startup")
    async def startup():
        """Initialize components on startup."""
        initialize_components()
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )

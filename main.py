

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import router
from config.setting import (API_HOST, API_PORT, DATA_PATH, OPENAI_LLM_MODEL_NAME,
                            EMBEDDING_MODEL_NAME, HF_TOKEN, META_LLM_MODEL_NAME,
                            MILVUS_COLLECTION, MILVUS_HOST, MILVUS_PORT,
                            OPENAI_API_KEY, VECTOR_DIM)
from data.loader import DataLoader
from embedding.embedder import TextEmbedder
from llm.llama_model import LlamaModel
from rag.retriever import ChromaRetriever
from llm.openai_model import OpenAIModel

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
    embedder = TextEmbedder(EMBEDDING_MODEL_NAME, OPENAI_API_KEY)
    
    # Initialize Milvus retriever
    retriever = ChromaRetriever(
        embedder=embedder
    )
    # Load FAQ data and set up the collection
    qa_pairs = data_loader.get_question_answer_pairs()
    retriever.setup_collection(qa_pairs)
    
    # Initialize LLM model
    llm_model = OpenAIModel(OPENAI_LLM_MODEL_NAME, OPENAI_API_KEY)
    
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

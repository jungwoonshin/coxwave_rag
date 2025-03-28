import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from config.setting import HISTORY_DIR
from utils.prompt import PromptBuilder

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[ChatMessage]] = None
    stream: bool = False
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    retrieved_docs: Optional[List[Dict[str, Any]]] = None

def save_chat_history(session_id: str, history: List[Dict[str, str]]):
    """
    Save chat history to a file.
    
    Args:
        session_id: Unique session identifier
        history: List of chat messages
    """
    if not session_id:
        return
    
    os.makedirs(HISTORY_DIR, exist_ok=True)
    history_file = os.path.join(HISTORY_DIR, f"{session_id}.json")
    
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

def get_chat_dependencies():
    """
    Get the dependencies for the chat endpoint.
    
    Returns:
        Tuple of (llm_model, retriever)
    """
    # These should be initialized elsewhere and passed here
    # For now, we'll assume they're available as global variables
    from main import llm_model, retriever
    return llm_model, retriever

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    dependencies: tuple = Depends(get_chat_dependencies)
):
    """
    Process a chat request and return a response.
    
    Args:
        request: Chat request with query and optional history
        dependencies: Tuple of (llm_model, retriever)
        
    Returns:
        Chat response
    """
    llm_model, retriever = dependencies
    
    # Convert pydantic model to dict for history
    history = [msg.dict() for msg in request.history] if request.history else []
    
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(request.query, top_k=3)
    
    # Check if query is relevant to the FAQ domain
    relevant = PromptBuilder.is_relevant([doc['score'] for doc in retrieved_docs])
    
    if not relevant:
        response = PromptBuilder.get_unrelated_message()
    else:
        # Generate response using RAG
        prompt_template = PromptBuilder.get_rag_prompt()
        response = llm_model.generate_rag_response(
            query=request.query,
            retrieved_docs=retrieved_docs,
            chat_history=history,
            prompt_template=prompt_template,
            stream=False
        )
    
    # Add the current query and response to history
    history.append({"role": "user", "content": request.query})
    history.append({"role": "assistant", "content": response})
    
    # Save history if session_id is provided
    if request.session_id:
        save_chat_history(request.session_id, history)
    
    return ChatResponse(
        response=response,
        retrieved_docs=retrieved_docs
    )

@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    dependencies: tuple = Depends(get_chat_dependencies)
):
    """
    Process a chat request and return a streaming response with proper Unicode support.
    """
    llm_model, retriever = dependencies
    
    # Convert pydantic model to dict for history
    history = [msg.dict() for msg in request.history] if request.history else []
    
    # Retrieve relevant documents
    retrieved_docs = retriever.retrieve(request.query, top_k=3)
    
    # Check if query is relevant to the FAQ domain
    relevant = PromptBuilder.is_relevant([doc['score'] for doc in retrieved_docs])
    
    async def event_generator():
        """Generate events for SSE streaming with proper Unicode handling."""
        if not relevant:
            # For irrelevant queries, just send the standard message
            message = PromptBuilder.get_unrelated_message()
            yield f"data: {json.dumps({'text': message}, ensure_ascii=False)}\n\n"
            return
            
        # Generate streaming response using RAG
        prompt_template = PromptBuilder.get_rag_prompt()
        full_response = ""
        
        # For Korean text, character-by-character streaming may be more appropriate
        # than trying to break at word boundaries
        buffer = ""
        
        for text_chunk in llm_model.generate_rag_response(
            query=request.query,
            retrieved_docs=retrieved_docs,
            chat_history=history,
            prompt_template=prompt_template,
            stream=True
        ):
            full_response += text_chunk
            buffer += text_chunk
            
            # Send complete sentences if possible
            if any(end_marker in buffer for end_marker in ['.', '?', '!', '\n', '。', '？', '！']):
                yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)
                buffer = ""
            
            # If buffer gets too large, send it anyway (even if not a complete sentence)
            elif len(buffer) >= 20:  # Adjust this number as needed
                yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.05)
                buffer = ""
        
        # Send any remaining content
        if buffer:
            yield f"data: {json.dumps({'text': buffer}, ensure_ascii=False)}\n\n"
        
        # Add the current query and response to history
        history.append({"role": "user", "content": request.query})
        history.append({"role": "assistant", "content": full_response})
        
        # Save history if session_id is provided
        if request.session_id:
            save_chat_history(request.session_id, history)
        
    return EventSourceResponse(event_generator())
# llm/llama_model.py
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import List, Dict, Any, Generator, Optional, Union
import logging
from threading import Thread

logger = logging.getLogger(__name__)


class LlamaModel:
    """
    Class for generating responses using the Meta Llama model.
    """
    def __init__(self, model_name: str, token: str):
        """
        Initialize the LlamaModel with a specific model.
        
        Args:
            model_name: Name of the model to use (e.g., "meta-llama/Llama-3.2-8B-Instruct")
            token: HuggingFace API token
        """
        self.model_name = model_name
        self.token = token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        try:
            # Try with trust_remote_code=True which is often needed for custom models
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=token,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loaded language model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            # Try with a different approach if the first one fails
            try:
                logger.info(f"Retrying with different parameters...")
                # Try with use_fast=False to fall back to Python tokenizer implementation
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    token=token,
                    trust_remote_code=True,
                    use_fast=False
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=token,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    trust_remote_code=True
                )
                logger.info(f"Successfully loaded model with fallback method: {model_name}")
            except Exception as fallback_error:
                logger.error(f"Error in fallback loading method: {fallback_error}")
                raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response from the LLM based on the given prompt.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (higher = more creative, lower = more deterministic)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the output tokens
            
        Returns:
            Generator yielding text chunks if stream=True, otherwise the complete response text
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        if stream:
            # Set up streamer for token-by-token generation
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            # Generate in a separate thread
            generation_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "streamer": streamer
            }
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield from the streamer
            for text in streamer:
                yield text
        else:
            # Generate the complete response at once
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,  # Correct parameter name
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            
            # Decode and return the complete response
            response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            return response
    
    def generate_rag_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        prompt_template: str = "",
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate a response using RAG retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents (question-answer pairs)
            chat_history: Optional chat history
            prompt_template: Template for the prompt
            stream: Whether to stream the output tokens
            
        Returns:
            Generator yielding text chunks if stream=True, otherwise the complete response text
        """
        # Format history if provided
        history_text = ""
        if chat_history and len(chat_history) > 0:
            for message in chat_history:
                if message["role"] == "user":
                    history_text += f"사용자: {message['content']}\n"
                else:
                    history_text += f"시스템: {message['content']}\n"
        
        # Format retrieved documents
        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"문서 {i+1}:\n"
            context += f"질문: {doc['question']}\n"
            context += f"답변: {doc['answer']}\n\n"
        
        # Fill the prompt template
        prompt = prompt_template.format(
            context=context,
            history=history_text,
            query=query
        )
        
        # Generate response
        return self.generate(prompt, stream=stream)
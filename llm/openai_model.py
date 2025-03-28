from openai import OpenAI
import logging
from typing import List, Dict, Any, Generator, Optional
import queue
import threading
import time

logger = logging.getLogger(__name__)

class OpenAIModel:
    """
    Class for generating responses using OpenAI models like GPT-4o-mini.
    """
    def __init__(self, model_name: str, api_key: str):
        """
        Initialize the OpenAIModel with a specific model.
        
        Args:
            model_name: Name of the model to use (e.g., "gpt-4o-mini")
            api_key: OpenAI API key
        """
        self.model_name = model_name
        self.api_key = api_key
        
        try:
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False
    ):
        """
        Generate a response from the OpenAI model based on the given prompt.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (higher = more creative, lower = more deterministic)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the output tokens
            
        Returns:
            Generator yielding text chunks if stream=True, otherwise the complete response text
        """
        try:
            # Create messages for the chat completion
            messages = [{"role": "user", "content": prompt}]
            
            # Request completion from OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Return a generator for streaming responses
                def response_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return response_generator()
            else:
                # Return the complete response
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_rag_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
        prompt_template: str = "",
        stream: bool = False
    ):
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
    
    def generate_with_system_message(
        self,
        query: str,
        system_message: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False
    ):
        """
        Generate a response with a system message.
        
        Args:
            query: User query
            system_message: System message for the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the output
            
        Returns:
            Generated response or stream
        """
        try:
            # Create messages with system and user messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Request completion from OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Return a generator for streaming responses
                def response_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return response_generator()
            else:
                # Return the complete response
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_chat_history(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False
    ):
        """
        Generate a response with a full chat history.
        
        Args:
            messages: List of messages in the format [{"role": "user|system|assistant", "content": "message"}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the output
            
        Returns:
            Generated response or stream
        """
        try:
            # Request completion from OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                # Return a generator for streaming responses
                def response_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return response_generator()
            else:
                # Return the complete response
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
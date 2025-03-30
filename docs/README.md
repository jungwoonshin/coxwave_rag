# FAQ Answering System

This project implements a FAQ answering system using Meta's Llama 3.2-8B-Instruct model and text-embedding-model 3 for embeddings. It uses Chroma as a vector database for RAG (Retrieval-Augmented Generation) and FastAPI for the API endpoints.

## Features

- RAG-based FAQ answering using chroma as the vector database
- Uses HuggingFace libraries (no LangChain or similar frameworks)
- Streaming responses with Server-Sent Events (SSE)
- Fallback response for unrelated questions
- Follow-up question generation
- Chat history saving

## Project Structure

```
project/
├── api/                  # API related code
│   ├── __init__.py
│   └── router.py         # FastAPI router definitions
├── config/               # Configuration settings
│   ├── __init__.py
│   └── settings.py       # Configuration parameters
├── data/                 # Data loading utilities
│   ├── __init__.py
│   └── loader.py         # Data loader for FAQ data
├── embedding/            # Embedding related code
│   ├── __init__.py
│   └── embedder.py       # Text embedding class
├── llm/                  # Language model related code
│   ├── __init__.py
│   └── model.py          # Llama model class
├── rag/                  # RAG implementation
│   ├── __init__.py
│   └── retriever.py      # Chroma retriever class
├── utils/                # Utility functions
│   ├── __init__.py
│   └── prompt.py         # Prompt templates
├── main.py               # Main application entry point
├── initialize_db.py      # Script to initialize Milvus database
└── requirements.txt      # Python dependencies
```

## Prerequisites

1. Python 3.8+
2. Chroma database (can be run with Docker)
4. HuggingFace API token with access to required models
5. GPU recommended for faster inference

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/faq-answering-system.git
   cd faq-answering-system
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
   
3. Start the API server:
   ```bash
   python main.py
   ```

## API Usage

### Chat Endpoint (Non-streaming)

```
POST /api/chat
```

Request body:
```json
{
  "query": "스마트 스토어 배송은 얼마나 걸리나요?",
  "history": [
    {
      "role": "user",
      "content": "스마트 스토어는 무엇인가요?"
    },
    {
      "role": "assistant",
      "content": "스마트 스토어는 네이버에서 제공하는 온라인 쇼핑몰 플랫폼입니다."
    }
  ],
  "session_id": "user123"
}
```

Response:
```json
{
  "response": "스마트 스토어의 배송 기간은 상품에 따라 다릅니다. 일반적으로 결제 완료 후 1-3일 내에 배송이 시작되며, 배송은 보통 1-2일 정도 소요됩니다. 혹시 빠른 배송이 필요하신가요?",
  "retrieved_docs": [
    {
      "question": "배송 기간은 얼마나 걸리나요?",
      "answer": "결제 완료 후 1-3일 내에 배송이 시작되며, 배송은 1-2일 정도 소요됩니다.",
      "score": 0.15
    },
    // ... other docs
  ]
}
```

### Chat Endpoint (Streaming)

```
POST /api/chat/stream
```

Request body: Same as non-streaming endpoint, but you'll receive a stream of Server-Sent Events (SSE).

## Chat History

Chat history is automatically saved in the `history` directory with the session ID as the filename. Each file contains the complete conversation history in JSON format.

## Configuration

Edit `config/settings.py` to change configuration parameters like model names, Milvus connection details, etc.

## Adding Your Own FAQ Data

The system expects FAQ data in a pickle file at `dataset/data.pkl`. The format should be a dictionary where keys are questions and values are answers.

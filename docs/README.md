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
coxwave_rag/                      # RAG 기반 FAQ 답변 시스템 프로젝트 루트.
├── main.py                       # FastAPI 서버 초기화 및 RAG 시스템 실행 진입점.
├── initialize_db.py              # ChromaDB 데이터베이스 초기화 및 설정 스크립트.
├── find_q_from_a.py              # 답변 기반 질문 생성 스크립트 (데이터셋 확장용).
├── failure_analysis.py           # RAG 시스템 실패 분석 및 성능 평가 스크립트.
├── validate.py                   # 생성된 QA 쌍 품질 검증 스크립트.
├── requirements.txt              # 프로젝트 Python 패키지 의존성 목록.
├── .gitignore                    # Git 추적 제외 대상 설정 파일.
│
├── api/                          # 웹 API 관련 모듈 디렉토리.
│   ├── __init__.py               # api 패키지 초기화 파일.
│   └── router.py                 # FastAPI 라우터 정의 (/api 엔드포인트 처리, FAQ 서비스).
│
├── config/                       # 프로젝트 설정 파일 디렉토리.
│   ├── __init__.py               # config 패키지 초기화 파일.
│   └── setting.py                # API 키, 모델명 등 핵심 설정 값 정의.
│
├── data/                         # 데이터 로딩 및 처리 모듈 디렉토리.
│   ├── __init__.py               # data 패키지 초기화 파일.
│   └── loader.py                 # FAQ 데이터셋 로딩 기능.
│
├── dataset/                      # FAQ 데이터셋 및 검증 모듈 디렉토리.
│   ├── __init__.py               # dataset 패키지 초기화 파일.
│   ├── cleaned_data.json/pkl     # 정제된 FAQ 데이터 (JSON/Pickle).
│   ├── qa_dataset_generated.json/pkl # 생성된 QA 쌍 데이터 (JSON/Pickle).
│   ├── original_data.json        # 원본 FAQ 데이터 (JSON).
│   ├── faq_data.txt              # 원시 FAQ 텍스트 데이터.
│   ├── transform.py              # 원본 데이터 형식 변환 스크립트.
│   ├── validator_openai.py       # OpenAI 모델 기반 QA 품질 검증.
│   └── validator_llama.py        # Llama 모델 기반 QA 품질 검증.
│
├── embedding/                    # 텍스트 임베딩 모듈 디렉토리.
│   ├── __init__.py               # embedding 패키지 초기화 파일.
│   └── embedder.py               # 텍스트 벡터 변환 (임베딩) 기능.
│
├── llm/                          # LLM 관련 모듈 디렉토리.
│   ├── __init__.py               # llm 패키지 초기화 파일.
│   ├── openai_model.py           # OpenAI GPT 모델 연동 및 응답 생성.
│   └── llama_model.py            # Meta Llama 모델 연동 및 응답 생성.
│
├── preprocessing/                # 데이터 전처리 모듈 디렉토리.
│   └── remove_redundancy.py      # 데이터셋 중복/유사 내용 제거 기능.
│
├── question_from_answer/         # 답변 기반 질문 생성 모듈 디렉토리.
│   ├── __init__.py               # question_from_answer 패키지 초기화 파일.
│   └── generate.py               # 답변 분석 기반 질문 생성 기능.
│
├── rag/                          # RAG 관련 모듈 디렉토리.
│   ├── __init__.py               # rag 패키지 초기화 파일.
│   ├── retriever.py              # 벡터 DB 문서 검색 기본 기능.
│   └── cluster_retriever.py      # 클러스터링 기반 효율적 문서 검색 기능.
│
├── utils/                        # 유틸리티 함수 디렉토리.
│   ├── __init__.py               # utils 패키지 초기화 파일.
│   └── prompt.py                 # LLM 프롬프트 템플릿 정의 및 관리.
│
├── find_failures/                # RAG 실패 사례 식별/분석 모듈 디렉토리.
│
├── chroma_data/                  # ChromaDB 벡터 데이터 저장 디렉토리.
│
├── history/                      # 사용자 질의/응답 내역 저장 디렉토리.
│
├── embedding_cache/              # 임베딩 결과 캐시 저장 디렉토리 (성능 최적화).
│
├── clustering_cache/             # 클러스터링 결과 캐시 저장 디렉토리 (성능 개선).
│
├── output/                       # 시스템 실행 결과 및 로그 저장 디렉토리.
│
└── docs/                         # 프로젝트 문서화 파일 (API 문서, 가이드 등) 디렉
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

# Mental Health Support Chatbot (RAG-Based)

An AI-powered chatbot that provides mental health support using Retrieval-Augmented Generation (RAG). The system understands user intent, detects emotions, supports multiple languages, and retrieves relevant counseling responses from a vector database.

## Features

- **Multilingual Support**: Detects and translates 20+ languages
- **Intent Classification**: Distinguishes between greetings, questions, gratitude, and out-of-scope requests
- **Emotion Detection**: Classifies user emotions (joy, sadness, anger, fear, neutral)
- **RAG Pipeline**: Retrieves relevant mental health counseling responses from a vector database
- **Response Summarization**: Optionally summarizes long retrieved contexts
- **Session Management**: Maintains conversation history using Redis
- **Language Flexibility**: Translates user queries to English, processes them, and translates responses back

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Qdrant (cloud-hosted)
- **Embeddings**: BAAI/bge-large-en-v1.5 (Hugging Face)
- **LLM**: Google Gemini 2.5-Flash
- **Cache**: Redis
- **ML Models**: DistilBERT (emotion), Linear SVC (language detection)

## Project Structure

```
├── rag/
│   ├── generator.py          # Main response generation pipeline
│   ├── retriever.py          # Vector DB retrieval
│   ├── helper_models/        # Emotion, language, intent classifiers
│   └── prompts/              # System prompts for LLM
├── routes/
│   └── generation.py         # API endpoints
├── controllers/
│   └── history_controller.py # Session & chat history management
├── models/                   # Request/response schemas
├── Notebooks/                # Model training notebooks
└── main.py                   # FastAPI app entry point
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in:
```bash
GEMINI_API_KEY=your_api_key
QDRANT_CLUSTER_ENDPOINT=your_qdrant_url
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=mental_health
TOP_K=1
TOP_R=10
```

### 3. Start Services
```bash
# Terminal 1: Redis (for session history)
redis-server --port 6380

# Terminal 2: FastAPI server
python main.py
```

## API Usage

### Generate Response
```bash
POST /generate
Content-Type: application/json

{
  "text": "I'm feeling anxious",
  "session_id": "optional-uuid"
}
```

**Response:**
```json
{
  "answer": "Here are some strategies for managing anxiety...",
  "session_id": "uuid"
}
```

## How It Works

1. **Language Detection**: Identifies user's language; translates to English if needed
2. **Intent Classification**: Determines if user is asking a question, greeting, expressing gratitude, etc.
3. **Emotion Detection**: Classifies emotional state for context-aware responses
4. **RAG Retrieval**: Searches vector DB for top-K relevant counseling responses (only for mental health questions)
5. **Response Generation**: Uses Gemini LLM with retrieved context to generate personalized response
6. **Translation**: Translates response back to user's original language
7. **History Storage**: Saves conversation in Redis for session continuity

## Model Downloads

- [Language Detection Model](https://drive.google.com/file/d/1ID40tWg3Bk29CMOvScfqSFv4Pi703842/view?usp=drive_link)
- [Emotion Detection Model](https://drive.google.com/drive/folders/1yxoEbYlZ7TmfWgc8HAysyt6lssGXzhHT?usp=sharing)

## Performance

- **Language Detection**: 99.57% accuracy across 20 languages
- **Intent Classification**: Highly accurate on mental health domain
- **Response Quality**: Context-aware, empathetic responses grounded in real counseling data

---
title: RAG Mental Health Chatbot Backend
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---


# Mental Health Support Chatbot (RAG-Based)

An AI-powered chatbot that provides mental health support using Retrieval-Augmented Generation (RAG). The system understands user intent, detects emotions, supports multiple languages, and retrieves relevant counseling responses from a vector database.

## Features

- **Multilingual Support**: Detects and translates 20+ languages
- **Intent Classification**: Distinguishes between greetings, questions, gratitude, and out-of-scope requests
- **Emotion Detection**: Classifies user emotions into 6 classes (sadness, joy, love, anger, fear, surprise)
- **RAG Pipeline**: Retrieves relevant mental health counseling responses from a vector database
- **Response Summarization**: Optionally summarizes long retrieved contexts
- **Session Management**: Maintains conversation history using Redis
- **Language Flexibility**: Translates user queries to English, processes them, and translates responses back

## Tech Stack

- **Backend**: FastAPI
- **Vector DB**: Qdrant (cloud-hosted)
- **Embeddings**: BAAI/bge-large-en-v1.5 (Hugging Face)
- **LLMs** (via Groq):
  - **Response generation, translation, summarization**: `meta-llama/llama-4-scout-17b-16e-instruct` (larger model for higher-quality output)
  - **Intent classification**: `llama-3.1-8b-instant` — a separate, **lighter model** dedicated to the intent classifier, since intent labeling is a simpler task that doesn't need the larger model's capacity. This keeps classification fast and cheap while reserving the heavier model for the user-facing responses.
- **Cache**: Redis
- **ML Models**: DistilBERT (emotion), Linear SVC (language detection)

## Project Structure

```
├── main.py                       # FastAPI app entry point (lifespan + router wiring)
├── ENUMS/                        # Shared enums (languages, intents)
├── rag/
│   ├── generator.py              # Main response generation pipeline
│   ├── retriever.py              # Vector DB retrieval + cross-encoder reranking
│   ├── store_ds_in_vector_db.py  # Embed & upload the dataset to Qdrant
│   ├── preprocess_csv.py         # Dataset preprocessing
│   ├── get_data_locally.py       # Local dataset download helper
│   └── helper_models/
│       ├── emotion_classifier/   # DistilBERT emotion classifier
│       ├── language_detector/    # Linear SVC language detector
│       ├── intent_classifier/    # Intent classifier
│       ├── llm_caller/           # Groq LLM wrapper (response, translate, summarize, intent)
│       ├── Preprocessor/         # Text preprocessing utilities
│       ├── model_objs/           # Saved model artifacts (.pkl)
│       └── prompts/              # Prompt templates (intent, translator, summarizer)
├── deployment/                   # FastAPI app package (run from project root)
│   ├── routes/                   # API endpoints (base, generation, health)
│   ├── controllers/              # Session & chat history management (Redis)
│   └── models/                   # Request/response schemas (Pydantic)
└── Notebooks/                    # Model training & experimentation notebooks
    ├── EmotionClassifierModel/   # DistilBERT emotion model training
    ├── LanguageDetectorModel/    # Language detection model training
    ├── IntentClassification/     # Intent classification experiments
    └── RAG Notebook/             # Retrieval + embedding experiments
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

# LLMs (Groq) — intent uses a lighter model than the rest
GROQ_API_KEY=your_api_key
SIDE_MODEL=meta-llama/llama-4-scout-17b-16e-instruct   # response, translation, summarization
INTENT_CLASSIFICATION_MODEL=llama-3.1-8b-instant       # lighter model for intent only

# Retrieval summarization — set to false to skip the summarizer LLM call
# and pass raw retrieved references straight to the response generator
SUMMARIZE_RETRIEVALS=true


LANGUAGE_DETECTION_MODEL_PATH=your-path   # auto-downloaded from HuggingFace if not found locally
EMOTION_MODEL_PATH=your-path              # auto-downloaded from HuggingFace if not found locally
```

> **Model auto-download:** If the file at `LANGUAGE_DETECTION_MODEL_PATH` or `EMOTION_MODEL_PATH` does not exist locally, it is automatically downloaded from HuggingFace Hub ([`Abdellmohsennn/language_detector`](https://huggingface.co/Abdellmohsennn/language_detector) and [`Abdellmohsennn/final_mental_emotion_model`](https://huggingface.co/Abdellmohsennn/final_mental_emotion_model) respectively) and saved to the configured path. Make sure the parent directory exists and is writable.

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
  "query": "I'm feeling anxious",
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
5. **Response Generation**: Uses the response LLM with retrieved context to generate a personalized response
6. **Translation**: Translates response back to user's original language
7. **History Storage**: Saves conversation in Redis for session continuity

## Emotion Labels

The emotion classifier predicts one of 6 classes, with the following numerical mappings:

| Label | Emotion  |
|-------|----------|
| 0     | Sadness  |
| 1     | Joy      |
| 2     | Love     |
| 3     | Anger    |
| 4     | Fear     |
| 5     | Surprise |

## Model Downloads

- [Language Detection Model](https://drive.google.com/file/d/1XHIuYFL-ogLVQFL2bFIoAEPTM2WrsGar/view?usp=drive_link)
- [Emotion Detection Model](https://drive.google.com/drive/folders/1yxoEbYlZ7TmfWgc8HAysyt6lssGXzhHT?usp=sharing)

## Performance

- **Language Detection**: 99.57% accuracy across 20 languages
- **Intent Classification**: Highly accurate on mental health domain
- **Response Quality**: Context-aware, empathetic responses grounded in real counseling data
## Deployment
- **Frontend**: [frontend_link](https://aliabdelmonam.github.io/chatbot-frontend/)
- **Backend**: [backend_link](https://aliabdelmenam-rag-mental-health-chatbot.hf.space)
### Metrics used in Axiom Dashboard
- [METRICS.MD](METRICS.MD) 
### Note
- if the there was a problem in the frontend, paste the backend link in the `Backend API URL`, and the endpoint `\generate`
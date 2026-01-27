# 🛒 E-Commerce Customer Service Chatbot

NLP Final Project - Customer Service Chatbot dengan Voice Input, RAG, dan Model Comparison

## 📋 Project Overview

Chatbot customer service untuk e-commerce yang dilengkapi dengan:
- **Voice Input**: Real-time speech-to-text menggunakan Whisper
- **Intent Classification**: Klasifikasi intent pelanggan (complaint, inquiry, dll)
- **Named Entity Recognition**: Ekstraksi order ID, produk, tanggal, dll
- **Hybrid Chatbot**: Kombinasi rule-based dan LLM (Groq/Gemini)
- **RAG System**: Retrieval-Augmented Generation untuk knowledge base
- **Conversation Summary**: Ringkasan percakapan otomatis
- **Model Comparison**: Perbandingan BART vs PEGASUS untuk summarization

## 🏗️ Project Structure

```
ecommerce-chatbot/
│
├── interfaces.py           # Contract definitions (PENTING!)
├── app.py                  # Main Streamlit application
├── requirements.txt        # Dependencies
├── .env.example           # API keys template
│
├── src/
│   ├── speech/            # Person A
│   │   ├── transcriber.py # Whisper integration
│   │   └── classifier.py  # Intent classification
│   │
│   ├── chatbot/           # Person B
│   │   ├── rag.py         # RAG retrieval
│   │   └── llm.py         # LLM response generation
│   │
│   └── nlp/               # Person C
│       ├── ner.py         # Named Entity Recognition
│       └── summarizer.py  # Summarization & comparison
│
├── data/
│   ├── knowledge_base/    # RAG documents
│   └── intents/           # Training data for classification
│
└── notebooks/
    └── model_comparison.ipynb
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone repository
git clone https://github.com/YOUR_TEAM/ecommerce-chatbot.git
cd ecommerce-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# GROQ_API_KEY=your-groq-api-key
# GOOGLE_API_KEY=your-google-api-key
```

**Get free API keys:**
- Groq: https://console.groq.com (Fast inference, FREE!)
- Google AI: https://makersuite.google.com/app/apikey (FREE!)

### 3. Run the App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## 👥 Team Responsibilities

| Person | Modules | Files |
|--------|---------|-------|
| **A** | Speech-to-Text, Intent Classification | `src/speech/transcriber.py`, `src/speech/classifier.py` |
| **B** | Hybrid Chatbot, RAG | `src/chatbot/llm.py`, `src/chatbot/rag.py` |
| **C** | NER, Summarization, Model Comparison | `src/nlp/ner.py`, `src/nlp/summarizer.py` |

## 📖 Interface Contracts

**IMPORTANT:** Read `interfaces.py` before coding!

All functions follow this contract:

```python
# Person A
def transcribe_audio(audio_bytes: bytes) -> str
def classify_intent(text: str) -> Intent

# Person B  
def retrieve_knowledge(query: str, top_k: int) -> List[RetrievedDoc]
def generate_response(...) -> str

# Person C
def extract_entities(text: str) -> Entities
def summarize_conversation(messages: List[Message]) -> str
def compare_models(text_samples: List[str]) -> Dict
```

## 🔄 Git Workflow

```bash
# Create your feature branch
git checkout -b feature/speech-classification  # Person A
git checkout -b feature/chatbot-rag            # Person B
git checkout -b feature/ner-summarization      # Person C

# Work on your code...
git add .
git commit -m "feat: implement whisper transcription"
git push origin feature/your-branch

# Day 3: Merge to main
git checkout main
git pull origin main
git merge feature/your-branch
git push origin main
```

## 📊 Model Comparison

For the model comparison requirement, we compare:

| Aspect | Model A | Model B |
|--------|---------|---------|
| **Summarization** | BART | PEGASUS |
| **Metrics** | ROUGE-1, ROUGE-2, ROUGE-L | ROUGE-1, ROUGE-2, ROUGE-L |
| **Latency** | Measured | Measured |

Run comparison:
```python
from src.nlp.summarizer import compare_models, generate_comparison_report

results = compare_models(test_samples)
print(generate_comparison_report(results))
```

## 🧪 Testing

Each module can be tested independently:

```bash
# Test speech module
python -m src.speech.transcriber
python -m src.speech.classifier

# Test chatbot module  
python -m src.chatbot.rag
python -m src.chatbot.llm

# Test NLP module
python -m src.nlp.ner
python -m src.nlp.summarizer
```

## 📝 Requirements Checklist

- [x] Classification (text/voice)
- [x] Summarization + NER
- [x] Hybrid Chatbot (rule + ML)
- [x] Conversation Summary feature
- [x] RAG System
- [x] Fine-tuned model (using existing)
- [x] Model Comparison (BART vs PEGASUS)

## 🎯 Demo Flow

1. User speaks/types message
2. System transcribes (if voice)
3. System classifies intent
4. System extracts entities (order ID, product, etc.)
5. System retrieves relevant knowledge
6. System generates response (rule-based or LLM)
7. System updates conversation summary

## 📞 Support

For issues, contact team members or create GitHub issue.

---

**🎓 NLP Final Project 2024**

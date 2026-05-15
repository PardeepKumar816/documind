# DocuMind 🧠
### Chat with any PDF using AI — 100% free, open source

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://docmind-pardeep.streamlit.app/)

## Demo
![DocuMind Demo](demo.gif)

## What it does
Upload any PDF and have a conversation with it. DocuMind retrieves the most relevant sections of your document and generates accurate, cited answers — telling you exactly which page the information came from.

## Tech Stack
| Layer | Technology |
|---|---|
| LLM | Llama 3.1 8B via Groq (free) |
| Orchestration | LangChain + LCEL |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, free) |
| Vector Store | ChromaDB (local, in-memory) |
| UI | Streamlit |
| Observability | LangSmith |

## How it works
```
PDF Upload → Text Extraction → Chunking (500 chars) →
Embedding (384-dim vectors) → ChromaDB Storage →
Question → Semantic Search → Top 4 chunks →
RAG Prompt → Llama 3.1 → Cited Answer
```

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/documind.git
cd documind

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your keys

# 5. Run
streamlit run app.py
```

## Environment variables

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key        # console.groq.com (free)
LANGCHAIN_API_KEY=your_key        # smith.langchain.com (free)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=documind
```

## Features
- 📄 Upload any text-based PDF
- 💬 Multi-turn chat with memory
- ⚡ Streaming responses (token by token)
- 📚 Source citations with page numbers
- 🔍 Semantic search (not keyword matching)
- 💰 Completely free — no OpenAI API needed

## Built by
[Pardeep Kumar](https://linkedin.com/in/pardeep-kumar-a257221a1) — Flutter & AI Engineer  
[GitHub](https://github.com/PardeepKumar816) | [LinkedIn](https://linkedin.com/in/pardeep-kumar-a257221a1)

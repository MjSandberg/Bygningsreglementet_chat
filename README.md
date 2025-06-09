# Bygningsreglementet Chat Bot

A chat interface for querying the Danish Building Regulations (Bygningsreglementet) using RAG (Retrieval Augmented Generation) and LLM technology.

## Features

- Web scraping of Bygningsreglementet content
- Local data storage and caching
- Semantic search using FAISS and BM25
- Interactive chat interface using Dash
- Multilingual support through SentenceTransformer
- LLM-powered responses through OpenRouter API

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MjSandberg/Bygningsreglementet_chat
cd BygningReglament_Bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
```

## Usage

1. First time setup (scrapes and processes data):
```bash
python scraper.py
```

2. Run the chat interface:
```bash
python app.py
```

The application will be available at `http://localhost:8050`

## Project Structure

- `app.py` - Main Dash application and chat interface
- `scraper.py` - Web scraping and data processing
- `rag.py` - Retrieval and generation components
- `bygningsreglementet_data.json` - Cached content data
- `faiss_index.bin` - FAISS similarity search index
- `bm25.pkl` - BM25 search index

## Dependencies

- dash
- dash-bootstrap-components
- beautifulsoup4
- sentence-transformers
- faiss-cpu
- rank_bm25
- openai
- numpy
- requests
- torch

## Environment Variables

- `OPENROUTER_API_KEY` - Your OpenRouter API key for LLM access

## Notes

- The first run will take some time to scrape and process the data
- Subsequent runs will use cached data unless forced to rescrape
- The FAISS index and BM25 model are saved locally for faster startup

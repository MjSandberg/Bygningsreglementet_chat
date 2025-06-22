# Bygningsreglementet Chat Bot

A chat interface for querying the Danish Building Regulations (Bygningsreglementet) using an intelligent agentic RAG (Retrieval Augmented Generation) system.

## Features

### Core RAG Capabilities
- **Web scraping** of Bygningsreglementet content with robust error handling
- **Local data storage** and caching with configurable paths
- **Hybrid search** using FAISS (semantic) and BM25 (keyword-based) retrieval
- **Interactive chat interface** using Dash with responsive design
- **Multilingual support** through SentenceTransformer models
- **LLM-powered responses** through OpenRouter API

### Agentic
- **Query Routing** - Automatically determines the best approach for each query
- **Web Search** - Searches external sources when local knowledge is insufficient
- **Context Sufficiency Evaluation** - Ensures adequate information before generating responses
- **Multi-Agent Coordination** - Orchestrates specialized agents for optimal results


## Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/MjSandberg/Bygningsreglementet_chat
cd Bygningsreglementet_chat
```

2. **Install dependencies:**
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

3. **Set up environment variables:**
```bash
# Edit .env with your OpenRouter API key
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

4. **Run the application:**
```bash
# Using the installed script
python main.py

# Or directly
python -m src.ui.app
```

The application will be available at `http://localhost:8050`

## Usage

### First Time Setup

If no data exists, the application will automatically scrape the data on first run. You can also manually scrape:

```bash
# Scrape data manually
python scraper_cli.py

# Force rescraping even if data exists
python scraper_cli.py --force

# Enable debug logging
python scraper_cli.py --debug
```

### Running the Chat Interface

```bash
# Run with default settings
python main.py

# Run with debug mode
DEBUG=true python main.py

# Run on different port
PORT=8080 python main.py
```

## Project Structure

```
├── src/                          # Main source code
│   ├── config.py                 # Configuration management
│   ├── scraper/                  # Web scraping components
│   │   ├── __init__.py
│   │   └── web_scraper.py        # Main scraper class
│   ├── rag/                      # RAG system components
│   │   ├── __init__.py
│   │   ├── retriever.py          # Document retrieval
│   │   └── generator.py          # Response generation
│   ├── ui/                       # User interface
│   │   ├── __init__.py
│   │   └── app.py                # Dash application
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── logging.py            # Logging configuration
│       └── text_processing.py   # Text processing utilities
├── data/                         # Data storage (created automatically)
│   ├── bygningsreglementet_data.json
│   ├── faiss_index.bin
│   └── bm25.pkl
├── logs/                         # Application logs (created automatically)
├── main.py                       # Main application entry point
├── scraper_cli.py               # CLI scraper script
├── pyproject.toml               # Project configuration
├── .env                         # Environment variables
└── README.md                    # This file
```

## Configuration

The application uses environment variables for configuration:

- `OPENROUTER_API_KEY` - Your OpenRouter API key (required)
- `DEBUG` - Enable debug mode (default: false)
- `PORT` - Application port (default: 8050)
- `HOST` - Application host (default: 127.0.0.1)

Additional configuration can be modified in `src/config.py`.

## Architecture

### Agentic RAG System

The system uses an intelligent multi-agent architecture that automatically determines the best approach for each query:

1. **Orchestrator Agent** - Central coordinator that analyzes queries and manages agent interactions
2. **Knowledge Retriever Agent** - Searches the local building regulations database
3. **Web Search Agent** - Searches external sources when local knowledge is insufficient
4. **Context Evaluator Agent** - Determines if gathered information is sufficient for response generation
5. **Generator Agent** - Creates responses with source attribution and quality validation

### Traditional RAG Components

The underlying RAG system combines:

1. **Semantic Search** (FAISS) - Uses sentence transformers for semantic similarity
2. **Keyword Search** (BM25) - Traditional keyword-based search
3. **Hybrid Scoring** - Combines both approaches with configurable weights
4. **LLM Generation** - Uses OpenRouter API for response generation

### Error Handling

- Comprehensive error handling throughout the application
- Graceful degradation when components fail
- Detailed logging for debugging and monitoring
- User-friendly error messages in the UI

## Development

### Testing

```bash
# Test the agentic system
python test_agentic_system.py

# Run type checking
mypy src/

# Run linting
ruff check src/

# Run formatting
ruff format src/
```

## Dependencies

Core dependencies:
- `dash` - Web application framework
- `dash-bootstrap-components` - UI components
- `beautifulsoup4` - Web scraping
- `sentence-transformers` - Semantic embeddings
- `faiss-cpu` - Vector similarity search
- `rank-bm25` - Keyword search
- `openai` - LLM API client
- `python-dotenv` - Environment variable management

## Troubleshooting

### Common Issues

1. **Missing API Key**: Ensure `OPENROUTER_API_KEY` is set in `.env`
2. **Port Already in Use**: Change the port with `PORT=8080 python main.py`
3. **Memory Issues**: Reduce chunk size in `src/config.py`
4. **Slow Startup**: Indices are being created; subsequent runs will be faster

### Logs

Check the logs in the `logs/` directory for detailed error information.

## License

This project is licensed under the MIT License.

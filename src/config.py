"""Configuration management for the Bygningsreglementet Chat Bot."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # API Configuration
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")
    DEFAULT_MODEL: str = "google/gemma-3-27b-it:free"
    
    # File paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DATA_FILE: Path = DATA_DIR / "bygningsreglementet_data.json"
    FAISS_INDEX_FILE: Path = DATA_DIR / "faiss_index.bin"
    BM25_FILE: Path = DATA_DIR / "bm25.pkl"
    
    # Model configuration
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Text processing
    CHUNK_SIZE: int = 2048
    CHUNK_OVERLAP: int = 200
    
    # Retrieval parameters
    DEFAULT_K: int = 5
    FAISS_WEIGHT: float = 0.75
    BM25_WEIGHT: float = 0.25
    MIN_SCORE: float = 0.4
    
    # Generation parameters
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500
    
    # App configuration
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8050"))
    HOST: str = os.getenv("HOST", "127.0.0.1")
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Ensure data directory exists
        cls.DATA_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_data_file_path(cls) -> str:
        """Get the data file path as string."""
        return str(cls.DATA_FILE)
    
    @classmethod
    def get_faiss_index_path(cls) -> str:
        """Get the FAISS index file path as string."""
        return str(cls.FAISS_INDEX_FILE)
    
    @classmethod
    def get_bm25_path(cls) -> str:
        """Get the BM25 file path as string."""
        return str(cls.BM25_FILE)

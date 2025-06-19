"""Test configuration module."""

import pytest
from pathlib import Path
import os
import tempfile

# Add src to path for testing
import sys
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.config import Config


def test_config_defaults():
    """Test that config has expected default values."""
    assert Config.DEFAULT_MODEL == "google/gemma-2-9b-it:free"
    assert Config.CHUNK_SIZE == 2048
    assert Config.CHUNK_OVERLAP == 200
    assert Config.DEFAULT_K == 5
    assert Config.FAISS_WEIGHT == 0.75
    assert Config.BM25_WEIGHT == 0.25
    assert Config.MIN_SCORE == 0.4
    assert Config.TEMPERATURE == 0.7
    assert Config.MAX_TOKENS == 500


def test_config_paths():
    """Test that config paths are properly constructed."""
    assert isinstance(Config.BASE_DIR, Path)
    assert isinstance(Config.DATA_DIR, Path)
    assert isinstance(Config.DATA_FILE, Path)
    assert isinstance(Config.FAISS_INDEX_FILE, Path)
    assert isinstance(Config.BM25_FILE, Path)
    
    # Test path methods return strings
    assert isinstance(Config.get_data_file_path(), str)
    assert isinstance(Config.get_faiss_index_path(), str)
    assert isinstance(Config.get_bm25_path(), str)


def test_config_validation_missing_api_key():
    """Test that validation fails when API key is missing."""
    # Temporarily remove API key
    original_key = Config.OPENROUTER_API_KEY
    Config.OPENROUTER_API_KEY = None
    
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY environment variable is required"):
        Config.validate()
    
    # Restore original key
    Config.OPENROUTER_API_KEY = original_key


def test_config_validation_success():
    """Test that validation succeeds with valid config."""
    # Ensure API key is set
    if not Config.OPENROUTER_API_KEY:
        Config.OPENROUTER_API_KEY = "test_key"
    
    # Should not raise any exception
    Config.validate()


if __name__ == "__main__":
    pytest.main([__file__])

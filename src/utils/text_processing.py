"""Text processing utilities."""

import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import Config


def fix_text(text: str) -> str:
    """
    Fix text formatting issues.
    
    Args:
        text: Raw text to fix
        
    Returns:
        Cleaned and formatted text
    """
    # Fix punctuation
    text = re.sub(r'\.(\S)', r'. \1', text)
    
    # Add space between numbers and letters
    text = re.sub(r'(\d)([a-zA-ZæøåÆØÅ])', r'\1 \2', text)
    
    # Add space between letters and numbers
    text = re.sub(r'([a-zA-ZæøåÆØÅ])(\d)', r'\1 \2', text)
    
    # Fix spaces around section symbols
    text = re.sub(r'§(\S)', r'§ \1', text)
    text = re.sub(r'(\S)§', r'\1 §', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter with configured parameters.
    
    Returns:
        Configured text splitter instance
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )


def split_text_if_needed(text: str, header: str, min_length: int = 250) -> List[str]:
    """
    Split text into chunks if it's longer than minimum length.
    
    Args:
        text: Text content to potentially split
        header: Header to prepend to each chunk
        min_length: Minimum length before splitting
        
    Returns:
        List of text chunks with headers
    """
    if len(text) > min_length:
        text_splitter = create_text_splitter()
        content_list = text_splitter.split_text(text)
        return [f"{header}: {chunk}" for chunk in content_list]
    else:
        return [f"{header}: {text}"]

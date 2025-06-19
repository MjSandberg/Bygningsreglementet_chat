"""Generator component for RAG system."""

from typing import List, Optional
import openai
import time
from functools import wraps

from ..config import Config
from ..utils.logging import get_logger
from .retriever import Retriever


class GenerationError(Exception):
    """Exception raised when text generation fails."""
    pass


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger = get_logger("generator")
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


class Generator:
    """Generator component for creating responses using LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            api_key: OpenRouter API key
            model: Model name to use for generation
            
        Raises:
            ValueError: If API key is not provided
        """
        self.logger = get_logger("generator")
        self.api_key = api_key or Config.OPENROUTER_API_KEY
        self.model = model or Config.DEFAULT_MODEL
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        self.logger.info(f"Initialized generator with model: {self.model}")

    @timing_decorator
    def generate_answer(
        self, 
        query: str, 
        data: List[str], 
        retriever: Retriever,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate an answer for a query using retrieved context.
        
        Args:
            query: User query
            data: Full dataset (for compatibility, not used directly)
            retriever: Retriever instance for getting relevant context
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
            
        Raises:
            GenerationError: If generation fails
        """
        try:
            # DEBUG: Log all inputs
            self.logger.info(f"=== GENERATOR DEBUG INFO ===")
            self.logger.info(f"User query: '{query}'")
            self.logger.info(f"Query type: {type(query)}")
            self.logger.info(f"Data type: {type(data)}")
            self.logger.info(f"Data length: {len(data) if data else 'None'}")
            if data:
                self.logger.info(f"First data item preview: {str(data[0])[:100]}...")
            self.logger.info(f"Retriever type: {type(retriever)}")
            self.logger.info(f"Temperature: {temperature}")
            self.logger.info(f"Max tokens: {max_tokens}")
            
            # Retrieve relevant documents
            retrieved_docs = retriever.retrieve(query)
            context = "\n".join(retrieved_docs)
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents for context")
            self.logger.info(f"Context preview: {context[:200]}...")
            self.logger.info(f"Full context length: {len(context)} characters")
            
            # Create prompt
            prompt = self._create_prompt(query, context)
            
            # Generate response
            response = self._call_llm(
                prompt, 
                temperature or Config.TEMPERATURE,
                max_tokens or Config.MAX_TOKENS
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            raise GenerationError(f"Failed to generate answer: {e}")

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = (
            f"Brugeren stiller dig et spørgsmål, du får givet en kontekst der minder "
            f"semantisk om brugerens spørgsmål og muligvis kan hjælpe dig med at give "
            f"et fyldestgørende svar:\n\n"
            f"# Kontekst: {context}\n\n"
            f"# Spørgsmål: {query}"
        )
        return prompt

    def _call_llm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """
        Call the LLM API.
        
        Args:
            prompt: Formatted prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response
            
        Raises:
            GenerationError: If API call fails
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                model=self.model,
            )
            
            response = chat_completion.choices[0].message.content
            if response is None:
                raise GenerationError("Received empty response from LLM")
                
            return response
            
        except openai.OpenAIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise GenerationError(f"API call failed: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM call: {e}")
            raise GenerationError(f"Unexpected error: {e}")

    def generate_simple_answer(
        self, 
        query: str, 
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate an answer with provided context (without retrieval).
        
        Args:
            query: User query
            context: Context to use for generation
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated answer
            
        Raises:
            GenerationError: If generation fails
        """
        try:
            prompt = self._create_prompt(query, context)
            response = self._call_llm(
                prompt,
                temperature or Config.TEMPERATURE,
                max_tokens or Config.MAX_TOKENS
            )
            return response
        except Exception as e:
            self.logger.error(f"Error in simple generation: {e}")
            raise GenerationError(f"Simple generation failed: {e}")

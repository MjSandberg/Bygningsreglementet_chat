"""Retriever component for RAG system."""

from typing import List, Dict, Optional
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import torch

from ..config import Config
from ..utils.logging import get_logger

# Set environment variables for torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ModelInitializationError(Exception):
    """Exception raised when model initialization fails."""
    pass


class Retriever:
    """Retriever component for semantic and keyword-based search."""
    
    def __init__(self, data: List[str], model_name: Optional[str] = None):
        """
        Initialize the retriever.
        
        Args:
            data: List of text passages to index
            model_name: Name of the sentence transformer model to use
            
        Raises:
            ModelInitializationError: If model initialization fails
        """
        self.logger = get_logger("retriever")
        self.data = data
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.index_file = Config.get_faiss_index_path()
        self.bm25_file = Config.get_bm25_path()
        
        # Initialize model
        self.emb_model = self._init_model()
        if self.emb_model is None:
            raise ModelInitializationError("Failed to initialize embedding model")
            
        # Load or create indices
        if os.path.exists(self.index_file) and os.path.exists(self.bm25_file):
            self.logger.info("Loading existing index and BM25...")
            self.load_index_and_bm25()
        else:
            self.logger.info("Creating new index and BM25...")
            self.create_and_save_index()

    def _init_model(self) -> Optional[SentenceTransformer]:
        """
        Initialize the sentence transformer model.
        
        Returns:
            SentenceTransformer model or None if initialization fails
        """
        try:
            self.logger.info(f"Initializing model: {self.model_name}")
            model = SentenceTransformer(self.model_name)
            return model
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            return None

    def create_embeddings(self) -> np.ndarray:
        """
        Create embeddings for all data chunks.
        
        Returns:
            Array of embeddings
            
        Raises:
            RuntimeError: If embedding creation fails
        """
        self.logger.info(f"Creating embeddings for {len(self.data)} chunks")
        embeddings = []
        
        for i, chunk in enumerate(self.data, 1):
            if i % 100 == 0:
                self.logger.info(f"Processing chunk {i}/{len(self.data)}")
            
            try:
                embedding = self.emb_model.encode(chunk, convert_to_numpy=True)
                embeddings.append(embedding)
            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {e}")
                continue
            
        if not embeddings:
            raise RuntimeError("No embeddings were created successfully")
            
        return np.array(embeddings)

    def create_and_save_index(self) -> None:
        """
        Create and save FAISS index and BM25.
        
        Raises:
            RuntimeError: If index creation fails
        """
        try:
            # Create embeddings
            embeddings = self.create_embeddings()
            
            # Create and save FAISS index
            self.logger.info("Creating FAISS index...")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
            
            self.logger.info("Saving FAISS index...")
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            
            # Create and save BM25
            self.logger.info("Creating and saving BM25...")
            self.bm25 = BM25Okapi([doc.split() for doc in self.data])
            with open(self.bm25_file, 'wb') as f:
                pickle.dump(self.bm25, f)
                
        except Exception as e:
            self.logger.error(f"Error in create_and_save_index: {e}")
            raise RuntimeError(f"Failed to create and save index: {e}")

    def load_index_and_bm25(self) -> None:
        """
        Load existing FAISS index and BM25.
        
        Raises:
            RuntimeError: If loading fails
        """
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.bm25_file, 'rb') as f:
                self.bm25 = pickle.load(f)
            self.logger.info("Successfully loaded existing indices")
        except Exception as e:
            self.logger.error(f"Error loading index and BM25: {e}")
            raise RuntimeError(f"Failed to load indices: {e}")

    def retrieve(
        self, 
        query: str, 
        k: Optional[int] = None,
        faiss_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        min_score: Optional[float] = None
    ) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            faiss_weight: Weight for FAISS scores
            bm25_weight: Weight for BM25 scores
            min_score: Minimum score threshold
            
        Returns:
            List of relevant document passages
            
        Raises:
            RuntimeError: If retrieval fails
        """
        # Use config defaults if not provided
        k = k or Config.DEFAULT_K
        faiss_weight = faiss_weight or Config.FAISS_WEIGHT
        bm25_weight = bm25_weight or Config.BM25_WEIGHT
        min_score = min_score or Config.MIN_SCORE
        
        try:
            # Get query embedding
            query_embedding = self.emb_model.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)

            # FAISS search
            faiss_distances, faiss_indices = self.index.search(
                query_embedding.astype('float32'), k
            )
            faiss_scores_norm = self._normalize_scores(
                1 - (faiss_distances[0] ** 2) / 2
            )

            # BM25 search
            bm25_scores = self.bm25.get_scores(query.split())
            bm25_scores_norm = self._normalize_scores(bm25_scores)

            # Combine scores
            combined_scores = self._combine_scores(
                faiss_indices[0], faiss_scores_norm, bm25_scores_norm, 
                faiss_weight, bm25_weight
            )

            # Filter by minimum score and return documents
            relevant_docs = [
                self.data[i] for i, score in combined_scores.items() 
                if score >= min_score
            ]
            
            self.logger.debug(f"Retrieved {len(relevant_docs)} documents for query")
            return relevant_docs
            
        except Exception as e:
            self.logger.error(f"Error in retrieve: {e}")
            raise RuntimeError(f"Retrieval failed: {e}")

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to 0-1 range.
        
        Args:
            scores: Array of scores to normalize
            
        Returns:
            Normalized scores
        """
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    def _combine_scores(
        self, 
        faiss_indices: np.ndarray, 
        faiss_scores_norm: np.ndarray, 
        bm25_scores_norm: np.ndarray, 
        faiss_weight: float, 
        bm25_weight: float
    ) -> Dict[int, float]:
        """
        Combine FAISS and BM25 scores.
        
        Args:
            faiss_indices: Indices from FAISS search
            faiss_scores_norm: Normalized FAISS scores
            bm25_scores_norm: Normalized BM25 scores
            faiss_weight: Weight for FAISS scores
            bm25_weight: Weight for BM25 scores
            
        Returns:
            Dictionary mapping document indices to combined scores
        """
        # Get top BM25 indices
        bm25_top_indices = np.argsort(bm25_scores_norm)[::-1][:len(faiss_indices)]
        
        # Combine indices
        combined_indices = set(faiss_indices).union(set(bm25_top_indices))
        
        # Calculate combined scores
        combined_scores = {}
        for idx in combined_indices:
            faiss_score = (
                faiss_scores_norm[list(faiss_indices).index(idx)] 
                if idx in faiss_indices else 0
            )
            bm25_score = bm25_scores_norm[idx]
            combined_scores[idx] = faiss_weight * faiss_score + bm25_weight * bm25_score
        
        return combined_scores

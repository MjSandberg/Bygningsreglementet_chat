from typing import List, Tuple
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
import time
import os
import pickle
import torch
import multiprocessing
from functools import partial

# Set environment variables for torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def init_model(model_name):
    """Initialize the model in a separate process"""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

class Retriever:
    def __init__(self, data: List[str], model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        self.data = data
        self.model_name = model_name
        self.index_file = "faiss_index.bin"
        self.bm25_file = "bm25.pkl"
        
        print("Initializing retriever...")
        
        # Initialize model in a separate process
        self.emb_model = init_model(model_name)
            
        if self.emb_model is None:
            raise RuntimeError("Failed to initialize model")
            
        if os.path.exists(self.index_file) and os.path.exists(self.bm25_file):
            print("Loading existing index and BM25...")
            self.load_index_and_bm25()
        else:
            print("Creating new index and BM25...")
            self.create_and_save_index()

    def create_embeddings(self):
        """Create embeddings using the natural chunks from input data"""
        print(f"Creating embeddings for {len(self.data)} chunks")
        embeddings = []
        
        for i, chunk in enumerate(self.data, 1):
            print(f"Processing chunk {i}/{len(self.data)}")
            try:
                embedding = self.emb_model.encode(chunk, convert_to_numpy=True)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                continue
            
        return np.array(embeddings)

    def create_and_save_index(self):
        """Create and save FAISS index and BM25"""
        try:
            # Create embeddings
            embeddings = self.create_embeddings()
            
            # Create and save FAISS index
            print("Creating FAISS index...")
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings.astype('float32'))
            
            print("Saving FAISS index...")
            faiss.write_index(self.index, self.index_file)
            
            # Create and save BM25
            print("Creating and saving BM25...")
            self.bm25 = BM25Okapi([doc.split() for doc in self.data])
            with open(self.bm25_file, 'wb') as f:
                pickle.dump(self.bm25, f)
        except Exception as e:
            print(f"Error in create_and_save_index: {e}")
            raise

    def load_index_and_bm25(self):
        """Load existing FAISS index and BM25"""
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.bm25_file, 'rb') as f:
                self.bm25 = pickle.load(f)
        except Exception as e:
            print(f"Error loading index and BM25: {e}")
            raise

    def retrieve(self, query: str, k: int = 5, faiss_weight: float = 0.75, bm25_weight: float = 0.25, min_score: float = 0.4) -> List[str]:
        try:
            query_embedding = self.emb_model.encode(query, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)

            faiss_distances, faiss_indices = self.index.search(query_embedding.astype('float32'), k)
            faiss_scores_norm = self._normalize_scores(1 - (faiss_distances[0] ** 2) / 2)

            bm25_scores = self.bm25.get_scores(query.split())
            bm25_scores_norm = self._normalize_scores(bm25_scores)

            combined_scores = self._combine_scores(faiss_indices[0], faiss_scores_norm, bm25_scores_norm, faiss_weight, bm25_weight)

            return [self.data[i] for i, score in combined_scores.items() if score >= min_score]
        except Exception as e:
            print(f"Error in retrieve: {e}")
            return []

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        min_score, max_score = np.min(scores), np.max(scores)
        return (scores - min_score) / (max_score - min_score) if max_score - min_score != 0 else np.zeros_like(scores)

    def _combine_scores(self, faiss_indices: np.ndarray, faiss_scores_norm: np.ndarray, bm25_scores_norm: np.ndarray, faiss_weight: float, bm25_weight: float) -> dict:
        combined_indices = set(faiss_indices).union(set(np.argsort(bm25_scores_norm)[::-1][:len(faiss_indices)]))
        combined_scores = {idx: faiss_weight * (faiss_scores_norm[list(faiss_indices).index(idx)] if idx in faiss_indices else 0) + bm25_weight * bm25_scores_norm[idx] for idx in combined_indices}
        return combined_scores

class Generator:
    def __init__(self, api_key: str, model: str = "google/gemma-2-9b-it:free"):
        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    @staticmethod
    def timing_decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
            return result
        return wrapper

    @timing_decorator
    def generate_answer(self, query: str, data: List[str], retriever: Retriever) -> str:
        retrieved_docs = retriever.retrieve(query)
        context = "\n".join(retrieved_docs)
        prompt = f"Brugeren stiller dig et spørgsmål, du får givet en kontekst der minder semantisk om brugerens spørgsmål og kan muligvis kan hjælpe dig med at give et fyldestgørende svar:\nKontekst: {context}\n\nSpørgsmål: {query}\n\nSvar:"

        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            model=self.model,
        )
        return chat_completion.choices[0].message.content

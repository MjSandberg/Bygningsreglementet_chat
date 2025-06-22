"""Knowledge Retriever Agent that wraps the existing RAG retriever."""

from typing import List, Optional
import numpy as np

from ..rag.retriever import Retriever
from ..utils.logging import get_logger
from .orchestrator import AgentResponse


class KnowledgeRetrieverAgent:
    """Agent that handles local knowledge retrieval from the building regulations."""
    
    def __init__(self, retriever: Retriever):
        """
        Initialize the knowledge retriever agent.
        
        Args:
            retriever: The RAG retriever instance
        """
        self.logger = get_logger("knowledge_retriever_agent")
        self.retriever = retriever
        
    def retrieve_knowledge(self, query: str, k: Optional[int] = None) -> Optional[AgentResponse]:
        """
        Retrieve knowledge from the local building regulations database.
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            AgentResponse with retrieved content and confidence score
        """
        try:
            self.logger.info(f"Retrieving knowledge for query: {query}")
            
            # Use the existing retriever to get relevant documents
            retrieved_docs = self.retriever.retrieve(query, k=k)
            
            if not retrieved_docs:
                self.logger.warning("No documents retrieved")
                return None
            
            # Calculate confidence based on retrieval quality
            confidence = self._calculate_confidence(query, retrieved_docs)
            
            # Combine retrieved documents
            content = "\n\n".join(retrieved_docs)
            
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents with confidence {confidence:.2f}")
            
            return AgentResponse(
                content=content,
                confidence=confidence,
                metadata={
                    "num_docs": len(retrieved_docs),
                    "source": "local_knowledge",
                    "retrieval_method": "hybrid_faiss_bm25",
                    "individual_docs": retrieved_docs  # Store individual documents for citation
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge: {e}")
            return None
    
    def _calculate_confidence(self, query: str, retrieved_docs: List[str]) -> float:
        """
        Calculate confidence score for the retrieved documents.
        
        Args:
            query: Original query
            retrieved_docs: List of retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        # Base confidence on number of documents and query-document similarity
        base_confidence = min(len(retrieved_docs) / 5.0, 1.0)  # More docs = higher confidence
        
        # Check for keyword overlap
        query_words = set(query.lower().split())
        total_overlap = 0
        
        for doc in retrieved_docs:
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            total_overlap += overlap
        
        # Normalize overlap score
        if query_words:
            overlap_score = min(total_overlap / (len(query_words) * len(retrieved_docs)), 1.0)
        else:
            overlap_score = 0.0
        
        # Combine scores
        confidence = (base_confidence * 0.6) + (overlap_score * 0.4)
        
        return min(confidence, 1.0)
    
    def assess_query_coverage(self, query: str) -> dict:
        """
        Assess how well the local knowledge base can cover the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with coverage assessment
        """
        try:
            # Try retrieval with different k values
            docs_k3 = self.retriever.retrieve(query, k=3)
            docs_k10 = self.retriever.retrieve(query, k=10)
            
            assessment = {
                "can_answer": len(docs_k3) > 0,
                "confidence": self._calculate_confidence(query, docs_k3),
                "num_relevant_docs": len(docs_k3),
                "extended_context_available": len(docs_k10) > len(docs_k3),
                "needs_web_search": len(docs_k3) == 0 or self._calculate_confidence(query, docs_k3) < 0.3
            }
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing query coverage: {e}")
            return {
                "can_answer": False,
                "confidence": 0.0,
                "num_relevant_docs": 0,
                "extended_context_available": False,
                "needs_web_search": True
            }
    
    def get_related_topics(self, query: str, max_topics: int = 5) -> List[str]:
        """
        Get related topics from the knowledge base.
        
        Args:
            query: User query
            max_topics: Maximum number of topics to return
            
        Returns:
            List of related topic strings
        """
        try:
            # Retrieve more documents to find related topics
            docs = self.retriever.retrieve(query, k=max_topics * 2)
            
            if not docs:
                return []
            
            # Extract topic indicators (section headers, key terms)
            topics = []
            for doc in docs[:max_topics]:
                # Look for section headers or key terms
                lines = doc.split('\n')
                for line in lines[:3]:  # Check first few lines
                    if line.strip() and len(line) < 100:  # Likely a header
                        if line not in topics:
                            topics.append(line.strip())
                            if len(topics) >= max_topics:
                                break
                if len(topics) >= max_topics:
                    break
            
            return topics
            
        except Exception as e:
            self.logger.error(f"Error getting related topics: {e}")
            return []

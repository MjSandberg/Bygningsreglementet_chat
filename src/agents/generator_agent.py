"""Generator Agent that wraps the existing RAG generator."""

from typing import List, Dict, Any, Optional

from ..rag.generator import Generator
from ..utils.logging import get_logger
from .orchestrator import AgentResponse


class GeneratorAgent:
    """Agent that handles response generation using gathered context."""
    
    def __init__(self, generator: Generator):
        """
        Initialize the generator agent.
        
        Args:
            generator: The RAG generator instance
        """
        self.logger = get_logger("generator_agent")
        self.generator = generator
    
    def generate_response(self, query: str, gathered_info: List[Dict[str, Any]]) -> str:
        """
        Generate a response using all gathered context information.
        
        Args:
            query: Original user query
            gathered_info: List of information gathered from different sources
            
        Returns:
            Generated response string
        """
        try:
            self.logger.info(f"Generating response for query: {query}")
            
            # Combine all gathered information into context
            combined_context = self._combine_gathered_info(gathered_info)
            
            if not combined_context:
                return self._generate_no_context_response(query)
            
            # Generate response using the existing generator
            response = self.generator.generate_simple_answer(
                query=query,
                context=combined_context
            )
            
            # Enhance response with source attribution
            enhanced_response = self._enhance_with_sources(response, gathered_info)
            
            self.logger.info("Response generated successfully")
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "Beklager, der opstod en fejl under generering af svaret. Prøv venligst igen."
    
    def _combine_gathered_info(self, gathered_info: List[Dict[str, Any]]) -> str:
        """
        Combine information from different sources into a coherent context.
        
        Args:
            gathered_info: List of information from different sources
            
        Returns:
            Combined context string
        """
        if not gathered_info:
            return ""
        
        context_parts = []
        
        # Sort by source priority (local knowledge first, then web search)
        sorted_info = sorted(
            gathered_info, 
            key=lambda x: self._get_source_priority(x.get("source", "unknown"))
        )
        
        for i, info in enumerate(sorted_info, 1):
            response = info.get("response")
            source = info.get("source", "unknown")
            
            if not response:
                continue
            
            content = getattr(response, 'content', '')
            confidence = getattr(response, 'confidence', 0.0)
            
            if content:
                # Format the content with source information
                source_label = self._get_source_label(source)
                context_part = f"=== {source_label} (Tillid: {confidence:.1f}) ===\n{content}"
                context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _get_source_priority(self, source: str) -> int:
        """
        Get priority order for sources (lower number = higher priority).
        
        Args:
            source: Source type string
            
        Returns:
            Priority number
        """
        priorities = {
            "local_knowledge": 1,
            "web_search": 2,
            "unknown": 3
        }
        return priorities.get(source, 3)
    
    def _get_source_label(self, source: str) -> str:
        """
        Get human-readable label for source type.
        
        Args:
            source: Source type string
            
        Returns:
            Human-readable label
        """
        labels = {
            "local_knowledge": "Bygningsreglementet Database",
            "web_search": "Web Søgning",
            "unknown": "Ukendt Kilde"
        }
        return labels.get(source, "Ukendt Kilde")
    
    def _enhance_with_sources(self, response: str, gathered_info: List[Dict[str, Any]]) -> str:
        """
        Enhance the response with source attribution.
        
        Args:
            response: Generated response
            gathered_info: Information sources used
            
        Returns:
            Enhanced response with source information
        """
        if not gathered_info:
            return response
        
        # Add source information at the end
        sources_used = []
        for info in gathered_info:
            source = info.get("source", "unknown")
            response_obj = info.get("response")
            
            if response_obj:
                confidence = getattr(response_obj, 'confidence', 0.0)
                metadata = getattr(response_obj, 'metadata', {})
                
                source_info = {
                    "label": self._get_source_label(source),
                    "confidence": confidence,
                    "metadata": metadata
                }
                sources_used.append(source_info)
        
        if sources_used:
            # Add source attribution
            source_text = "\n\n---\n**Kilder:**\n"
            for i, source_info in enumerate(sources_used, 1):
                label = source_info["label"]
                confidence = source_info["confidence"]
                source_text += f"{i}. {label} (Tillid: {confidence:.1f})\n"
            
            response += source_text
        
        return response
    
    def _generate_no_context_response(self, query: str) -> str:
        """
        Generate a response when no context is available.
        
        Args:
            query: Original query
            
        Returns:
            No-context response
        """
        return (
            "Jeg kunne ikke finde tilstrækkelig information til at besvare dit spørgsmål "
            "i bygningsreglementet eller gennem web-søgning. "
            "Prøv venligst at omformulere dit spørgsmål eller være mere specifik om, "
            "hvilken del af bygningsreglementet du er interesseret i."
        )
    
    def generate_summary_response(self, query: str, gathered_info: List[Dict[str, Any]]) -> str:
        """
        Generate a summary response highlighting key points from multiple sources.
        
        Args:
            query: Original user query
            gathered_info: List of information gathered from different sources
            
        Returns:
            Summary response string
        """
        try:
            if not gathered_info:
                return self._generate_no_context_response(query)
            
            # Create a summary-focused context
            summary_context = self._create_summary_context(gathered_info)
            
            # Create a summary-focused prompt
            summary_query = f"Lav et sammendrag af følgende information relateret til spørgsmålet '{query}'"
            
            response = self.generator.generate_simple_answer(
                query=summary_query,
                context=summary_context
            )
            
            return self._enhance_with_sources(response, gathered_info)
            
        except Exception as e:
            self.logger.error(f"Error generating summary response: {e}")
            return self.generate_response(query, gathered_info)  # Fallback to regular response
    
    def _create_summary_context(self, gathered_info: List[Dict[str, Any]]) -> str:
        """
        Create a context optimized for summary generation.
        
        Args:
            gathered_info: List of information from different sources
            
        Returns:
            Summary-optimized context string
        """
        context_parts = []
        
        for info in gathered_info:
            response = info.get("response")
            source = info.get("source", "unknown")
            
            if not response:
                continue
            
            content = getattr(response, 'content', '')
            if content:
                # Extract key points (first few sentences or paragraphs)
                key_points = self._extract_key_points(content)
                source_label = self._get_source_label(source)
                
                context_part = f"Fra {source_label}:\n{key_points}"
                context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _extract_key_points(self, content: str, max_length: int = 500) -> str:
        """
        Extract key points from content for summary.
        
        Args:
            content: Full content text
            max_length: Maximum length of extracted text
            
        Returns:
            Key points text
        """
        if len(content) <= max_length:
            return content
        
        # Split into sentences and take the first few
        sentences = content.split('. ')
        key_text = ""
        
        for sentence in sentences:
            if len(key_text + sentence) <= max_length:
                key_text += sentence + ". "
            else:
                break
        
        return key_text.strip()
    
    def validate_response_quality(self, query: str, response: str, gathered_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate the quality of the generated response.
        
        Args:
            query: Original query
            response: Generated response
            gathered_info: Information used for generation
            
        Returns:
            Quality assessment dictionary
        """
        try:
            assessment = {
                "length_appropriate": 50 <= len(response) <= 2000,
                "addresses_query": self._check_query_addressed(query, response),
                "uses_sources": len(gathered_info) > 0,
                "has_attribution": "Kilder:" in response,
                "confidence_score": self._calculate_response_confidence(gathered_info)
            }
            
            assessment["overall_quality"] = sum([
                assessment["length_appropriate"],
                assessment["addresses_query"],
                assessment["uses_sources"],
                assessment["has_attribution"]
            ]) / 4.0
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error validating response quality: {e}")
            return {"overall_quality": 0.5, "error": str(e)}
    
    def _check_query_addressed(self, query: str, response: str) -> bool:
        """
        Check if the response addresses the query.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            True if query appears to be addressed
        """
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words
        stop_words = {"og", "eller", "i", "på", "til", "af", "for", "med", "er", "det", "en", "et"}
        query_words = query_words - stop_words
        
        if not query_words:
            return True  # Can't evaluate empty query
        
        overlap = len(query_words.intersection(response_words))
        return overlap / len(query_words) >= 0.3  # At least 30% overlap
    
    def _calculate_response_confidence(self, gathered_info: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence for the response based on sources.
        
        Args:
            gathered_info: Information used for generation
            
        Returns:
            Confidence score between 0 and 1
        """
        if not gathered_info:
            return 0.0
        
        total_confidence = 0.0
        total_weight = 0.0
        
        for info in gathered_info:
            response = info.get("response")
            if response:
                confidence = getattr(response, 'confidence', 0.5)
                source = info.get("source", "unknown")
                
                # Weight by source reliability
                weight = 1.0 if source == "local_knowledge" else 0.7
                
                total_confidence += confidence * weight
                total_weight += weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0

"""Citation Agent that provides reliable citations for generated responses."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.logging import get_logger


@dataclass
class Citation:
    """Represents a citation with source information."""
    id: str
    source_type: str  # "local_knowledge" or "web_search"
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    confidence: float = 0.0
    relevance_score: float = 0.0
    excerpt: str = ""  # Relevant excerpt from the content


@dataclass
class CitationMapping:
    """Maps parts of the response to specific citations."""
    response_text: str
    citations: List[Citation]
    citation_mappings: List[Dict[str, Any]]  # Maps response segments to citation IDs


class CitationAgent:
    """Agent that handles citation generation and mapping for responses."""
    
    def __init__(self):
        """Initialize the citation agent."""
        self.logger = get_logger("citation_agent")
    
    def generate_citations(self, response: str, gathered_info: List[Dict[str, Any]]) -> CitationMapping:
        """
        Generate citations for a response based on gathered information.
        
        Args:
            response: The generated response text
            gathered_info: List of information gathered from different sources
            
        Returns:
            CitationMapping with citations and mappings
        """
        try:
            self.logger.info("Generating citations for response")
            
            # Extract citations from gathered info
            citations = self._extract_citations(gathered_info)
            
            # Map response segments to citations
            citation_mappings = self._map_response_to_citations(response, citations)
            
            # Create citation mapping
            citation_mapping = CitationMapping(
                response_text=response,
                citations=citations,
                citation_mappings=citation_mappings
            )
            
            self.logger.info(f"Generated {len(citations)} citations with {len(citation_mappings)} mappings")
            return citation_mapping
            
        except Exception as e:
            self.logger.error(f"Error generating citations: {e}")
            # Return empty citation mapping on error
            return CitationMapping(
                response_text=response,
                citations=[],
                citation_mappings=[]
            )
    
    def _extract_citations(self, gathered_info: List[Dict[str, Any]]) -> List[Citation]:
        """
        Extract citations from gathered information.
        
        Args:
            gathered_info: List of information from different sources
            
        Returns:
            List of Citation objects
        """
        citations = []
        citation_id = 1
        
        for info in gathered_info:
            source = info.get("source", "unknown")
            response_obj = info.get("response")
            
            if not response_obj:
                continue
            
            content = getattr(response_obj, 'content', '')
            confidence = getattr(response_obj, 'confidence', 0.0)
            metadata = getattr(response_obj, 'metadata', {})
            
            if not content:
                continue
            
            # Check if we have individual documents (for local knowledge)
            individual_docs = metadata.get("individual_docs", [])
            individual_results = metadata.get("individual_results", [])
            
            if source == "local_knowledge" and individual_docs:
                # Create separate citations for each document
                for doc in individual_docs:
                    if doc.strip():  # Skip empty documents
                        citation = Citation(
                            id=f"cite_{citation_id}",
                            source_type=source,
                            content=doc,
                            confidence=confidence,
                            relevance_score=confidence,
                            title=self._extract_title_from_doc(doc),
                            excerpt=self._extract_excerpt(doc, max_length=200)
                        )
                        citations.append(citation)
                        citation_id += 1
            elif source == "web_search" and individual_results:
                # Create separate citations for each web search result
                for result in individual_results:
                    if result.get("snippet", "").strip():  # Skip empty results
                        citation = Citation(
                            id=f"cite_{citation_id}",
                            source_type=source,
                            content=result.get("snippet", ""),
                            confidence=confidence,
                            relevance_score=confidence,
                            title=result.get("title", "Web Search Result"),
                            url=result.get("url", ""),
                            excerpt=self._extract_excerpt(result.get("snippet", ""), max_length=200)
                        )
                        citations.append(citation)
                        citation_id += 1
            else:
                # Create single citation for the source
                citation = Citation(
                    id=f"cite_{citation_id}",
                    source_type=source,
                    content=content,
                    confidence=confidence,
                    relevance_score=confidence
                )
                
                # Add source-specific information
                if source == "local_knowledge":
                    citation.title = metadata.get("title", "Bygningsreglementet")
                    citation.excerpt = self._extract_excerpt(content, max_length=200)
                    
                elif source == "web_search":
                    citation.title = metadata.get("title", "Web Search Result")
                    citation.url = metadata.get("url", "")
                    citation.excerpt = self._extract_excerpt(content, max_length=200)
                
                citations.append(citation)
                citation_id += 1
        
        return citations
    
    def _extract_title_from_doc(self, doc: str) -> str:
        """
        Extract a title from a document.
        
        Args:
            doc: Document content
            
        Returns:
            Extracted title or default
        """
        lines = doc.split('\n')
        
        # Look for the first non-empty line that looks like a title
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and len(line) < 150:  # Reasonable title length
                # Remove common prefixes
                if line.startswith('Section ') or line.startswith('Kapitel '):
                    return line
                elif ':' in line:
                    # Take the part before the colon as title
                    title = line.split(':')[0].strip()
                    if len(title) > 10:  # Reasonable title length
                        return title
                elif len(line) > 10:  # First substantial line
                    return line
        
        # Fallback to generic title
        return "Bygningsreglementet Dokument"
    
    def _extract_excerpt(self, content: str, max_length: int = 200) -> str:
        """
        Extract a relevant excerpt from content.
        
        Args:
            content: Full content text
            max_length: Maximum length of excerpt
            
        Returns:
            Relevant excerpt
        """
        if len(content) <= max_length:
            return content
        
        # Try to find a good breaking point (sentence end)
        sentences = content.split('. ')
        excerpt = ""
        
        for sentence in sentences:
            if len(excerpt + sentence) <= max_length:
                excerpt += sentence + ". "
            else:
                break
        
        if not excerpt:
            # Fallback to simple truncation
            excerpt = content[:max_length] + "..."
        
        return excerpt.strip()
    
    def _map_response_to_citations(self, response: str, citations: List[Citation]) -> List[Dict[str, Any]]:
        """
        Map parts of the response to specific citations.
        
        Args:
            response: Generated response text
            citations: Available citations
            
        Returns:
            List of mappings between response segments and citations
        """
        mappings = []
        
        if not citations:
            return mappings
        
        # Split response into sentences for mapping
        sentences = self._split_into_sentences(response)
        
        for i, sentence in enumerate(sentences):
            # Find the most relevant citation for this sentence
            best_citation = self._find_best_citation_for_sentence(sentence, citations)
            
            if best_citation:
                mapping = {
                    "sentence_index": i,
                    "sentence": sentence,
                    "citation_id": best_citation.id,
                    "relevance_score": best_citation.relevance_score
                }
                mappings.append(mapping)
        
        return mappings
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (could be enhanced with more sophisticated NLP)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _find_best_citation_for_sentence(self, sentence: str, citations: List[Citation]) -> Optional[Citation]:
        """
        Find the best citation for a given sentence.
        
        Args:
            sentence: Sentence to find citation for
            citations: Available citations
            
        Returns:
            Best matching citation or None
        """
        if not citations:
            return None
        
        best_citation = None
        best_score = 0.0
        
        # Convert sentence to lowercase for comparison
        sentence_lower = sentence.lower()
        sentence_words = set(sentence_lower.split())
        
        for citation in citations:
            # Calculate similarity score based on word overlap
            citation_words = set(citation.content.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(sentence_words.intersection(citation_words))
            union = len(sentence_words.union(citation_words))
            
            if union > 0:
                jaccard_score = intersection / union
                
                # Weight by citation confidence
                weighted_score = jaccard_score * citation.confidence
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_citation = citation
        
        # Only return citation if score is above threshold
        if best_score > 0.1:  # Minimum threshold for relevance
            return best_citation
        
        return None
    
    def format_citations_for_display(self, citation_mapping: CitationMapping) -> Dict[str, Any]:
        """
        Format citations for display in the UI.
        
        Args:
            citation_mapping: Citation mapping to format
            
        Returns:
            Formatted citation data for UI display
        """
        try:
            formatted_citations = []
            
            for citation in citation_mapping.citations:
                formatted_citation = {
                    "id": citation.id,
                    "source_type": citation.source_type,
                    "title": citation.title or "Unknown Source",
                    "excerpt": citation.excerpt,
                    "confidence": round(citation.confidence, 2),
                    "relevance_score": round(citation.relevance_score, 2)
                }
                
                # Add source-specific fields
                if citation.source_type == "web_search" and citation.url:
                    formatted_citation["url"] = citation.url
                
                formatted_citations.append(formatted_citation)
            
            # Create sentence-to-citation mappings for highlighting
            sentence_mappings = {}
            for mapping in citation_mapping.citation_mappings:
                sentence_idx = mapping["sentence_index"]
                citation_id = mapping["citation_id"]
                
                if sentence_idx not in sentence_mappings:
                    sentence_mappings[sentence_idx] = []
                
                sentence_mappings[sentence_idx].append({
                    "citation_id": citation_id,
                    "relevance_score": mapping["relevance_score"]
                })
            
            return {
                "citations": formatted_citations,
                "sentence_mappings": sentence_mappings,
                "total_citations": len(formatted_citations),
                "response_text": citation_mapping.response_text
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting citations for display: {e}")
            return {
                "citations": [],
                "sentence_mappings": {},
                "total_citations": 0,
                "response_text": citation_mapping.response_text
            }
    
    def generate_citation_text(self, citation_mapping: CitationMapping) -> str:
        """
        Generate formatted citation text to append to responses.
        
        Args:
            citation_mapping: Citation mapping
            
        Returns:
            Formatted citation text
        """
        try:
            if not citation_mapping.citations:
                return ""
            
            citation_text = "\n\n**Kilder:**\n"
            
            for i, citation in enumerate(citation_mapping.citations, 1):
                source_label = self._get_source_label(citation.source_type)
                confidence_text = f"(Tillid: {citation.confidence:.1f})"
                
                if citation.source_type == "web_search" and citation.url:
                    citation_line = f"{i}. [{citation.title}]({citation.url}) - {source_label} {confidence_text}"
                else:
                    citation_line = f"{i}. {citation.title} - {source_label} {confidence_text}"
                
                citation_text += citation_line + "\n"
            
            return citation_text
            
        except Exception as e:
            self.logger.error(f"Error generating citation text: {e}")
            return ""
    
    def _get_source_label(self, source_type: str) -> str:
        """
        Get human-readable label for source type.
        
        Args:
            source_type: Source type string
            
        Returns:
            Human-readable label
        """
        labels = {
            "local_knowledge": "Bygningsreglementet Database",
            "web_search": "Web Søgning",
            "unknown": "Ukendt Kilde"
        }
        return labels.get(source_type, "Ukendt Kilde")
    
    def validate_citations(self, citation_mapping: CitationMapping) -> Dict[str, Any]:
        """
        Validate the quality and completeness of citations.
        
        Args:
            citation_mapping: Citation mapping to validate
            
        Returns:
            Validation results
        """
        try:
            validation = {
                "has_citations": len(citation_mapping.citations) > 0,
                "citation_count": len(citation_mapping.citations),
                "mapped_sentences": len(citation_mapping.citation_mappings),
                "coverage_ratio": 0.0,
                "average_confidence": 0.0,
                "source_diversity": len(set(c.source_type for c in citation_mapping.citations))
            }
            
            # Calculate coverage ratio
            total_sentences = len(self._split_into_sentences(citation_mapping.response_text))
            if total_sentences > 0:
                validation["coverage_ratio"] = validation["mapped_sentences"] / total_sentences
            
            # Calculate average confidence
            if citation_mapping.citations:
                total_confidence = sum(c.confidence for c in citation_mapping.citations)
                validation["average_confidence"] = total_confidence / len(citation_mapping.citations)
            
            # Overall quality score
            validation["quality_score"] = (
                validation["coverage_ratio"] * 0.4 +
                validation["average_confidence"] * 0.4 +
                min(validation["source_diversity"] / 2, 1.0) * 0.2
            )
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Error validating citations: {e}")
            return {"quality_score": 0.0, "error": str(e)}

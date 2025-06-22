"""Context Evaluator Agent that determines if gathered information is sufficient."""

from typing import Dict, List, Optional, Any
import re

from ..utils.logging import get_logger
from .orchestrator import AgentResponse


class ContextEvaluatorAgent:
    """Agent that evaluates whether gathered context is sufficient to answer queries."""
    
    def __init__(self):
        """Initialize the context evaluator agent."""
        self.logger = get_logger("context_evaluator_agent")
        
        # Minimum thresholds for different types of queries
        self.confidence_thresholds = {
            "simple": 0.6,
            "complex": 0.7,
            "technical": 0.8,
            "regulatory": 0.75
        }
        
        # Keywords that indicate different query types
        self.query_type_indicators = {
            "technical": ["beregning", "dimensioner", "konstruktion", "statik", "styrke"],
            "regulatory": ["krav", "regel", "lovgivning", "bestemmelse", "paragraf"],
            "complex": ["hvordan", "proces", "procedure", "trin", "guide"],
            "simple": ["hvad", "hvor", "hvornår", "definition"]
        }
    
    def evaluate_context_sufficiency(
        self, 
        context: Dict[str, Any], 
        latest_response: Optional[AgentResponse] = None
    ) -> bool:
        """
        Evaluate if the gathered context is sufficient to answer the query.
        
        Args:
            context: Current context dictionary with query and gathered info
            latest_response: Latest agent response to evaluate
            
        Returns:
            True if context is sufficient, False otherwise
        """
        try:
            query = context.get("query", "")
            gathered_info = context.get("gathered_info", [])
            
            self.logger.info(f"Evaluating context sufficiency for query: {query}")
            
            # Basic checks
            if not gathered_info:
                self.logger.info("No information gathered yet")
                return False
            
            # Determine query type and required confidence threshold
            query_type = self._classify_query_type(query)
            required_confidence = self.confidence_thresholds.get(query_type, 0.7)
            
            self.logger.info(f"Query classified as '{query_type}', required confidence: {required_confidence}")
            
            # Evaluate information quality
            info_quality_score = self._evaluate_information_quality(gathered_info, query)
            
            # Evaluate coverage
            coverage_score = self._evaluate_query_coverage(query, gathered_info)
            
            # Evaluate source diversity
            source_diversity_score = self._evaluate_source_diversity(gathered_info)
            
            # Calculate overall sufficiency score
            overall_score = (
                info_quality_score * 0.5 +
                coverage_score * 0.3 +
                source_diversity_score * 0.2
            )
            
            is_sufficient = overall_score >= required_confidence
            
            self.logger.info(
                f"Context evaluation - Quality: {info_quality_score:.2f}, "
                f"Coverage: {coverage_score:.2f}, Diversity: {source_diversity_score:.2f}, "
                f"Overall: {overall_score:.2f}, Sufficient: {is_sufficient}"
            )
            
            return is_sufficient
            
        except Exception as e:
            self.logger.error(f"Error evaluating context sufficiency: {e}")
            return len(gathered_info) > 0  # Fallback to simple check
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify the query type based on keywords.
        
        Args:
            query: User query
            
        Returns:
            Query type string
        """
        query_lower = query.lower()
        
        # Check for each type in order of specificity
        for query_type, keywords in self.query_type_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type
        
        # Default classification based on query length and complexity
        if len(query.split()) > 10:
            return "complex"
        else:
            return "simple"
    
    def _evaluate_information_quality(self, gathered_info: List[Dict], query: str) -> float:
        """
        Evaluate the quality of gathered information.
        
        Args:
            gathered_info: List of information from different sources
            query: Original query
            
        Returns:
            Quality score between 0 and 1
        """
        if not gathered_info:
            return 0.0
        
        total_quality = 0.0
        total_weight = 0.0
        
        for info in gathered_info:
            response = info.get("response")
            if not response:
                continue
            
            # Base quality on confidence score
            confidence = getattr(response, 'confidence', 0.5)
            
            # Adjust based on source type
            source = info.get("source", "unknown")
            source_weight = self._get_source_weight(source)
            
            # Adjust based on content relevance
            content = getattr(response, 'content', '')
            relevance_score = self._calculate_content_relevance(query, content)
            
            # Calculate weighted quality
            quality = confidence * relevance_score
            weighted_quality = quality * source_weight
            
            total_quality += weighted_quality
            total_weight += source_weight
        
        return total_quality / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_query_coverage(self, query: str, gathered_info: List[Dict]) -> float:
        """
        Evaluate how well the gathered information covers the query.
        
        Args:
            query: Original query
            gathered_info: List of gathered information
            
        Returns:
            Coverage score between 0 and 1
        """
        if not gathered_info:
            return 0.0
        
        query_words = set(query.lower().split())
        
        # Remove common stop words
        stop_words = {"og", "eller", "i", "på", "til", "af", "for", "med", "er", "det", "en", "et"}
        query_words = query_words - stop_words
        
        if not query_words:
            return 0.5  # Neutral score if no meaningful words
        
        covered_words = set()
        
        for info in gathered_info:
            response = info.get("response")
            if response:
                content = getattr(response, 'content', '').lower()
                content_words = set(content.split())
                covered_words.update(query_words.intersection(content_words))
        
        coverage = len(covered_words) / len(query_words)
        return min(coverage, 1.0)
    
    def _evaluate_source_diversity(self, gathered_info: List[Dict]) -> float:
        """
        Evaluate the diversity of information sources.
        
        Args:
            gathered_info: List of gathered information
            
        Returns:
            Diversity score between 0 and 1
        """
        if not gathered_info:
            return 0.0
        
        sources = set()
        for info in gathered_info:
            source = info.get("source", "unknown")
            sources.add(source)
        
        # Score based on number of different sources
        if len(sources) == 1:
            return 0.5
        elif len(sources) == 2:
            return 0.8
        else:
            return 1.0
    
    def _get_source_weight(self, source: str) -> float:
        """
        Get weight for different source types.
        
        Args:
            source: Source type string
            
        Returns:
            Weight value
        """
        weights = {
            "local_knowledge": 1.0,  # High trust in local building regulations
            "web_search": 0.7,       # Medium trust in web search results
            "unknown": 0.5
        }
        
        return weights.get(source, 0.5)
    
    def _calculate_content_relevance(self, query: str, content: str) -> float:
        """
        Calculate how relevant content is to the query.
        
        Args:
            query: Original query
            content: Content to evaluate
            
        Returns:
            Relevance score between 0 and 1
        """
        if not content:
            return 0.0
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(content_words))
        relevance = overlap / len(query_words) if query_words else 0.0
        
        # Boost score if content is substantial
        if len(content) > 200:
            relevance *= 1.1
        
        return min(relevance, 1.0)
    
    def suggest_additional_searches(self, context: Dict[str, Any]) -> List[str]:
        """
        Suggest additional search queries if context is insufficient.
        
        Args:
            context: Current context dictionary
            
        Returns:
            List of suggested search queries
        """
        try:
            query = context.get("query", "")
            gathered_info = context.get("gathered_info", [])
            
            suggestions = []
            
            # Analyze what's missing
            query_words = set(query.lower().split())
            covered_words = set()
            
            for info in gathered_info:
                response = info.get("response")
                if response:
                    content = getattr(response, 'content', '').lower()
                    covered_words.update(query_words.intersection(set(content.split())))
            
            missing_words = query_words - covered_words
            
            # Generate specific search suggestions
            if missing_words:
                # Create more specific queries
                for word in list(missing_words)[:3]:  # Limit to 3 suggestions
                    specific_query = f"{word} bygningsreglement"
                    suggestions.append(specific_query)
            
            # Add context-specific suggestions
            query_lower = query.lower()
            if "krav" in query_lower and not any("krav" in info.get("source", "") for info in gathered_info):
                suggestions.append(f"{query} krav bestemmelser")
            
            if "eksempel" in query_lower:
                suggestions.append(f"{query} case study praksis")
            
            return suggestions[:3]  # Return max 3 suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating search suggestions: {e}")
            return []
    
    def get_context_gaps(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify gaps in the current context.
        
        Args:
            context: Current context dictionary
            
        Returns:
            Dictionary describing context gaps
        """
        try:
            query = context.get("query", "")
            gathered_info = context.get("gathered_info", [])
            
            gaps = {
                "missing_sources": [],
                "low_confidence_areas": [],
                "uncovered_query_terms": [],
                "suggestions": []
            }
            
            # Check for missing source types
            available_sources = {info.get("source") for info in gathered_info}
            if "local_knowledge" not in available_sources:
                gaps["missing_sources"].append("local_knowledge")
            if "web_search" not in available_sources:
                gaps["missing_sources"].append("web_search")
            
            # Check for low confidence responses
            for info in gathered_info:
                response = info.get("response")
                if response and getattr(response, 'confidence', 0) < 0.5:
                    gaps["low_confidence_areas"].append(info.get("source", "unknown"))
            
            # Check for uncovered query terms
            query_words = set(query.lower().split())
            covered_words = set()
            for info in gathered_info:
                response = info.get("response")
                if response:
                    content = getattr(response, 'content', '').lower()
                    covered_words.update(query_words.intersection(set(content.split())))
            
            gaps["uncovered_query_terms"] = list(query_words - covered_words)
            
            # Generate suggestions
            gaps["suggestions"] = self.suggest_additional_searches(context)
            
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error identifying context gaps: {e}")
            return {"missing_sources": [], "low_confidence_areas": [], "uncovered_query_terms": [], "suggestions": []}

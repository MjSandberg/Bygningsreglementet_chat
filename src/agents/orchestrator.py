"""Orchestrator agent that coordinates between different specialized agents."""

from typing import Dict, List, Optional, Any
from enum import Enum
import re

from ..utils.logging import get_logger


class QueryType(Enum):
    """Types of queries the orchestrator can handle."""
    LOCAL_KNOWLEDGE = "local_knowledge"
    WEB_SEARCH_NEEDED = "web_search_needed"
    COMPLEX_MULTI_STEP = "complex_multi_step"
    CLARIFICATION_NEEDED = "clarification_needed"


class AgentResponse:
    """Response from an agent."""
    
    def __init__(self, content: str, confidence: float, metadata: Optional[Dict] = None):
        self.content = content
        self.confidence = confidence
        self.metadata = metadata or {}


class Orchestrator:
    """Main orchestrator that coordinates between different agents."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.logger = get_logger("orchestrator")
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_context: Dict[str, Any] = {}
        
        # Agent instances will be injected
        self.knowledge_retriever = None
        self.web_search_agent = None
        self.context_evaluator = None
        self.generator_agent = None
    
    def set_agents(self, knowledge_retriever, web_search_agent, context_evaluator, generator_agent, citation_agent=None):
        """Set the agent instances."""
        self.knowledge_retriever = knowledge_retriever
        self.web_search_agent = web_search_agent
        self.context_evaluator = context_evaluator
        self.generator_agent = generator_agent
        self.citation_agent = citation_agent
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the agentic system.
        
        Args:
            query: User's question or request
            
        Returns:
            Generated response
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Analyze the query type
            query_type = self._analyze_query(query)
            self.logger.info(f"Query classified as: {query_type.value}")
            
            # Initialize context for this query
            self.current_context = {
                "query": query,
                "query_type": query_type,
                "gathered_info": [],
                "search_attempts": 0,
                "max_attempts": 3
            }
            
            # Process based on query type
            response = self._execute_query_strategy(query_type)
            
            # Add to conversation history
            self.conversation_history.append({
                "query": query,
                "response": response,
                "query_type": query_type.value,
                "context": self.current_context.copy()
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return "Beklager, der opstod en fejl under behandling af dit spørgsmål. Prøv venligst igen."
    
    def _analyze_query(self, query: str) -> QueryType:
        """
        Analyze the query to determine the best processing strategy.
        
        Args:
            query: User query
            
        Returns:
            QueryType indicating the processing strategy
        """
        query_lower = query.lower()
        
        # Keywords that suggest web search might be needed
        web_search_indicators = [
            "nyeste", "seneste", "opdateret", "ny", "2024", "2025", "aktuel",
            "sammenlign", "versus", "vs", "forskel", "alternativ",
            "eksempel", "case", "erfaring", "praksis"
        ]
        
        # Keywords that suggest complex multi-step processing
        complex_indicators = [
            "hvordan", "trin for trin", "proces", "procedure", "guide",
            "beregn", "udregn", "dimensioner", "design"
        ]
        
        # Check for web search indicators
        if any(indicator in query_lower for indicator in web_search_indicators):
            return QueryType.WEB_SEARCH_NEEDED
        
        # Check for complex processing indicators
        if any(indicator in query_lower for indicator in complex_indicators):
            return QueryType.COMPLEX_MULTI_STEP
        
        # Check if query is too vague
        if len(query.split()) < 3 or query.endswith("?") and len(query) < 20:
            return QueryType.CLARIFICATION_NEEDED
        
        # Default to local knowledge search
        return QueryType.LOCAL_KNOWLEDGE
    
    def _execute_query_strategy(self, query_type: QueryType) -> str:
        """
        Execute the appropriate strategy based on query type.
        
        Args:
            query_type: Type of query processing needed
            
        Returns:
            Generated response
        """
        if query_type == QueryType.CLARIFICATION_NEEDED:
            return self._handle_clarification()
        
        # Start with local knowledge retrieval
        local_response = self._try_local_knowledge()
        
        if query_type == QueryType.LOCAL_KNOWLEDGE:
            # Evaluate if local knowledge is sufficient
            if self._is_context_sufficient(local_response):
                return self._generate_final_response()
            else:
                # Fall back to web search
                return self._try_web_search_fallback()
        
        elif query_type == QueryType.WEB_SEARCH_NEEDED:
            # Try web search in addition to local knowledge
            web_response = self._try_web_search()
            return self._generate_final_response()
        
        elif query_type == QueryType.COMPLEX_MULTI_STEP:
            # Handle complex queries with iterative approach
            return self._handle_complex_query()
        
        return self._generate_final_response()
    
    def _try_local_knowledge(self) -> Optional[AgentResponse]:
        """
        Try to retrieve information from local knowledge base.
        
        Returns:
            AgentResponse or None if failed
        """
        try:
            if not self.knowledge_retriever:
                self.logger.warning("Knowledge retriever not available")
                return None
            
            self.logger.info("Attempting local knowledge retrieval")
            response = self.knowledge_retriever.retrieve_knowledge(
                self.current_context["query"]
            )
            
            if response:
                self.current_context["gathered_info"].append({
                    "source": "local_knowledge",
                    "response": response
                })
                self.logger.info(f"Local knowledge retrieved with confidence: {response.confidence}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in local knowledge retrieval: {e}")
            return None
    
    def _try_web_search(self) -> Optional[AgentResponse]:
        """
        Try to search the web for additional information.
        
        Returns:
            AgentResponse or None if failed
        """
        try:
            if not self.web_search_agent:
                self.logger.warning("Web search agent not available")
                return None
            
            self.logger.info("Attempting web search")
            response = self.web_search_agent.search_web(
                self.current_context["query"]
            )
            
            if response:
                self.current_context["gathered_info"].append({
                    "source": "web_search",
                    "response": response
                })
                self.logger.info(f"Web search completed with confidence: {response.confidence}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in web search: {e}")
            return None
    
    def _try_web_search_fallback(self) -> str:
        """
        Try web search as fallback when local knowledge is insufficient.
        
        Returns:
            Generated response
        """
        self.logger.info("Local knowledge insufficient, trying web search fallback")
        web_response = self._try_web_search()
        
        if web_response and self._is_context_sufficient(web_response):
            return self._generate_final_response()
        else:
            return self._generate_partial_response()
    
    def _handle_complex_query(self) -> str:
        """
        Handle complex multi-step queries.
        
        Returns:
            Generated response
        """
        self.logger.info("Handling complex query with iterative approach")
        
        # Try multiple information gathering attempts
        max_attempts = self.current_context["max_attempts"]
        
        for attempt in range(max_attempts):
            self.current_context["search_attempts"] = attempt + 1
            
            # Check if we have sufficient context
            if self._is_context_sufficient():
                break
            
            # Try web search for additional context
            if attempt > 0:  # First attempt already tried local knowledge
                self._try_web_search()
        
        return self._generate_final_response()
    
    def _handle_clarification(self) -> str:
        """
        Handle queries that need clarification.
        
        Returns:
            Clarification request
        """
        return (
            "Dit spørgsmål er lidt uklart. Kan du venligst være mere specifik? "
            "For eksempel, hvilken del af bygningsreglementet er du interesseret i, "
            "eller hvad er den konkrete situation du står i?"
        )
    
    def _is_context_sufficient(self, response: Optional[AgentResponse] = None) -> bool:
        """
        Evaluate if the gathered context is sufficient to answer the query.
        
        Args:
            response: Optional specific response to evaluate
            
        Returns:
            True if context is sufficient
        """
        try:
            if not self.context_evaluator:
                # Fallback evaluation
                return len(self.current_context["gathered_info"]) > 0
            
            return self.context_evaluator.evaluate_context_sufficiency(
                self.current_context, response
            )
            
        except Exception as e:
            self.logger.error(f"Error evaluating context sufficiency: {e}")
            return len(self.current_context["gathered_info"]) > 0
    
    def _generate_final_response(self) -> str:
        """
        Generate the final response using all gathered context.
        
        Returns:
            Generated response
        """
        try:
            if not self.generator_agent:
                return "Beklager, generator ikke tilgængelig."
            
            # Generate the response
            response = self.generator_agent.generate_response(
                self.current_context["query"],
                self.current_context["gathered_info"]
            )
            
            # Generate citations if citation agent is available
            if self.citation_agent:
                try:
                    citation_mapping = self.citation_agent.generate_citations(
                        response, self.current_context["gathered_info"]
                    )
                    
                    # Store citation mapping in context for UI access
                    self.current_context["citation_mapping"] = citation_mapping
                    
                    # Add citation text to response
                    citation_text = self.citation_agent.generate_citation_text(citation_mapping)
                    if citation_text:
                        response += citation_text
                        
                except Exception as e:
                    self.logger.error(f"Error generating citations: {e}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating final response: {e}")
            return "Beklager, der opstod en fejl under generering af svaret."
    
    def _generate_partial_response(self) -> str:
        """
        Generate a partial response when context is insufficient.
        
        Returns:
            Partial response with available information
        """
        if not self.current_context["gathered_info"]:
            return (
                "Jeg kunne ikke finde tilstrækkelig information til at besvare dit spørgsmål. "
                "Prøv venligst at omformulere dit spørgsmål eller være mere specifik."
            )
        
        return self._generate_final_response()
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        self.current_context.clear()

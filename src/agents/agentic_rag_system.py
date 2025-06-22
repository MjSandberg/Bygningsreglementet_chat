"""Agentic RAG System that coordinates all agents."""

from typing import List, Optional

from ..rag.retriever import Retriever
from ..rag.generator import Generator
from ..utils.logging import get_logger
from .orchestrator import Orchestrator
from .knowledge_retriever_agent import KnowledgeRetrieverAgent
from .web_search_agent import WebSearchAgent
from .context_evaluator_agent import ContextEvaluatorAgent
from .generator_agent import GeneratorAgent
from .citation_agent import CitationAgent


class AgenticRAGSystem:
    """Main agentic RAG system that coordinates all agents."""
    
    def __init__(self, data: List[str], api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the agentic RAG system.
        
        Args:
            data: List of text passages for the knowledge base
            api_key: OpenRouter API key
            model: Model name to use for generation
        """
        self.logger = get_logger("agentic_rag_system")
        
        # Initialize core RAG components
        self.retriever = Retriever(data)
        self.generator = Generator(api_key, model)
        
        # Initialize agents
        self.knowledge_retriever_agent = KnowledgeRetrieverAgent(self.retriever)
        self.web_search_agent = WebSearchAgent()
        self.context_evaluator_agent = ContextEvaluatorAgent()
        self.generator_agent = GeneratorAgent(self.generator)
        self.citation_agent = CitationAgent()
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator()
        self.orchestrator.set_agents(
            knowledge_retriever=self.knowledge_retriever_agent,
            web_search_agent=self.web_search_agent,
            context_evaluator=self.context_evaluator_agent,
            generator_agent=self.generator_agent,
            citation_agent=self.citation_agent
        )
        
        self.logger.info("Agentic RAG system initialized successfully")
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the agentic system.
        
        Args:
            query: User's question or request
            
        Returns:
            Generated response
        """
        try:
            self.logger.info(f"Processing query through agentic system: {query}")
            
            # Use the orchestrator to process the query
            response = self.orchestrator.process_query(query)
            
            self.logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query in agentic system: {e}")
            return "Beklager, der opstod en fejl under behandling af dit spørgsmål. Prøv venligst igen."
    
    def get_last_citation_mapping(self):
        """Get the citation mapping from the last processed query."""
        try:
            history = self.orchestrator.get_conversation_history()
            if history:
                last_entry = history[-1]
                context = last_entry.get("context", {})
                return context.get("citation_mapping")
            return None
        except Exception as e:
            self.logger.error(f"Error getting last citation mapping: {e}")
            return None
    
    def get_conversation_history(self):
        """Get the conversation history from the orchestrator."""
        return self.orchestrator.get_conversation_history()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.orchestrator.clear_conversation_history()
    
    def get_system_status(self) -> dict:
        """
        Get the status of all system components.
        
        Returns:
            Dictionary with system status information
        """
        try:
            status = {
                "retriever_ready": self.retriever is not None,
                "generator_ready": self.generator is not None,
                "knowledge_agent_ready": self.knowledge_retriever_agent is not None,
                "web_search_agent_ready": self.web_search_agent is not None,
                "context_evaluator_ready": self.context_evaluator_agent is not None,
                "generator_agent_ready": self.generator_agent is not None,
                "orchestrator_ready": self.orchestrator is not None,
                "conversation_history_length": len(self.orchestrator.get_conversation_history())
            }
            
            status["all_systems_ready"] = all([
                status["retriever_ready"],
                status["generator_ready"],
                status["knowledge_agent_ready"],
                status["web_search_agent_ready"],
                status["context_evaluator_ready"],
                status["generator_agent_ready"],
                status["orchestrator_ready"]
            ])
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "all_systems_ready": False}
    
    def test_agents(self, test_query: str = "Hvad er brandkrav for bygninger?") -> dict:
        """
        Test all agents with a simple query.
        
        Args:
            test_query: Query to use for testing
            
        Returns:
            Dictionary with test results
        """
        try:
            self.logger.info(f"Testing agents with query: {test_query}")
            
            test_results = {}
            
            # Test knowledge retriever agent
            try:
                knowledge_response = self.knowledge_retriever_agent.retrieve_knowledge(test_query)
                test_results["knowledge_retriever"] = {
                    "success": knowledge_response is not None,
                    "confidence": getattr(knowledge_response, 'confidence', 0.0) if knowledge_response else 0.0
                }
            except Exception as e:
                test_results["knowledge_retriever"] = {"success": False, "error": str(e)}
            
            # Test web search agent
            try:
                web_response = self.web_search_agent.search_web(test_query, max_results=2)
                test_results["web_search"] = {
                    "success": web_response is not None,
                    "confidence": getattr(web_response, 'confidence', 0.0) if web_response else 0.0
                }
            except Exception as e:
                test_results["web_search"] = {"success": False, "error": str(e)}
            
            # Test context evaluator
            try:
                test_context = {
                    "query": test_query,
                    "gathered_info": [],
                    "search_attempts": 0,
                    "max_attempts": 3
                }
                evaluation = self.context_evaluator_agent.evaluate_context_sufficiency(test_context)
                test_results["context_evaluator"] = {"success": True, "evaluation": evaluation}
            except Exception as e:
                test_results["context_evaluator"] = {"success": False, "error": str(e)}
            
            # Test generator agent
            try:
                test_info = []
                if test_results.get("knowledge_retriever", {}).get("success"):
                    test_info.append({
                        "source": "local_knowledge",
                        "response": knowledge_response
                    })
                
                if test_info:
                    generator_response = self.generator_agent.generate_response(test_query, test_info)
                    test_results["generator"] = {
                        "success": len(generator_response) > 0,
                        "response_length": len(generator_response)
                    }
                else:
                    test_results["generator"] = {"success": False, "error": "No test info available"}
            except Exception as e:
                test_results["generator"] = {"success": False, "error": str(e)}
            
            # Test orchestrator (full system test)
            try:
                orchestrator_response = self.orchestrator.process_query(test_query)
                test_results["orchestrator"] = {
                    "success": len(orchestrator_response) > 0,
                    "response_length": len(orchestrator_response)
                }
            except Exception as e:
                test_results["orchestrator"] = {"success": False, "error": str(e)}
            
            # Calculate overall success rate
            successful_tests = sum(1 for result in test_results.values() if result.get("success", False))
            total_tests = len(test_results)
            test_results["overall_success_rate"] = successful_tests / total_tests if total_tests > 0 else 0.0
            
            self.logger.info(f"Agent testing completed. Success rate: {test_results['overall_success_rate']:.2f}")
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Error testing agents: {e}")
            return {"error": str(e), "overall_success_rate": 0.0}
    
    def get_agent_statistics(self) -> dict:
        """
        Get statistics about agent usage and performance.
        
        Returns:
            Dictionary with agent statistics
        """
        try:
            history = self.orchestrator.get_conversation_history()
            
            stats = {
                "total_queries": len(history),
                "query_types": {},
                "average_sources_per_query": 0,
                "most_common_sources": {},
                "average_confidence": 0.0
            }
            
            if not history:
                return stats
            
            total_sources = 0
            total_confidence = 0.0
            confidence_count = 0
            
            for entry in history:
                # Count query types
                query_type = entry.get("query_type", "unknown")
                stats["query_types"][query_type] = stats["query_types"].get(query_type, 0) + 1
                
                # Count sources
                context = entry.get("context", {})
                gathered_info = context.get("gathered_info", [])
                total_sources += len(gathered_info)
                
                # Calculate confidence
                for info in gathered_info:
                    response = info.get("response")
                    if response:
                        confidence = getattr(response, 'confidence', 0.0)
                        total_confidence += confidence
                        confidence_count += 1
                        
                        # Count source types
                        source = info.get("source", "unknown")
                        stats["most_common_sources"][source] = stats["most_common_sources"].get(source, 0) + 1
            
            # Calculate averages
            if len(history) > 0:
                stats["average_sources_per_query"] = total_sources / len(history)
            
            if confidence_count > 0:
                stats["average_confidence"] = total_confidence / confidence_count
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting agent statistics: {e}")
            return {"error": str(e)}

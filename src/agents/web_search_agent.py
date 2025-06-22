"""Web Search Agent for finding external information."""

from typing import List, Optional, Dict, Any
import requests
import json
from bs4 import BeautifulSoup
import re
from urllib.parse import quote_plus

from ..utils.logging import get_logger
from .orchestrator import AgentResponse


class WebSearchAgent:
    """Agent that handles web searches for external information."""
    
    def __init__(self):
        """Initialize the web search agent."""
        self.logger = get_logger("web_search_agent")
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Danish building regulation related sites to prioritize
        self.priority_domains = [
            "bygningsreglementet.dk",
            "retsinformation.dk", 
            "boligstyrelsen.dk",
            "energistyrelsen.dk",
            "brs.dk",
            "bygherreforeningen.dk"
        ]
    
    def search_web(self, query: str, max_results: int = 5) -> Optional[AgentResponse]:
        """
        Search the web for information related to the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to process
            
        Returns:
            AgentResponse with search results and confidence
        """
        try:
            self.logger.info(f"Searching web for: {query}")
            
            # Enhance query for building regulations context
            enhanced_query = self._enhance_query_for_building_regulations(query)
            
            # Perform search using DuckDuckGo (no API key required)
            search_results = self._search_duckduckgo(enhanced_query, max_results)
            
            if not search_results:
                self.logger.warning("No search results found")
                return None
            
            # Process and extract content from results
            processed_content = self._process_search_results(search_results)
            
            if not processed_content:
                self.logger.warning("No content extracted from search results")
                return None
            
            # Calculate confidence based on result quality
            confidence = self._calculate_search_confidence(query, search_results, processed_content)
            
            self.logger.info(f"Web search completed with {len(search_results)} results, confidence: {confidence:.2f}")
            
            return AgentResponse(
                content=processed_content,
                confidence=confidence,
                metadata={
                    "num_results": len(search_results),
                    "source": "web_search",
                    "search_engine": "duckduckgo",
                    "query_enhanced": enhanced_query != query,
                    "individual_results": search_results  # Store individual results for citation
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in web search: {e}")
            return None
    
    def _enhance_query_for_building_regulations(self, query: str) -> str:
        """
        Enhance the query with building regulation specific terms.
        
        Args:
            query: Original query
            
        Returns:
            Enhanced query string
        """
        # Add Danish building regulation context
        building_terms = ["bygningsreglement", "bygningsreglementet", "BR18", "BR15"]
        
        # Check if query already contains building regulation terms
        query_lower = query.lower()
        has_building_terms = any(term.lower() in query_lower for term in building_terms)
        
        if not has_building_terms:
            # Add building regulation context
            enhanced = f"{query} bygningsreglement Danmark"
        else:
            enhanced = query
        
        return enhanced
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of search result dictionaries
        """
        try:
            # DuckDuckGo instant answer API (limited but free)
            search_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Extract abstract if available
            if data.get('Abstract'):
                results.append({
                    'title': data.get('AbstractText', 'DuckDuckGo Abstract'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', ''),
                    'source': 'duckduckgo_abstract'
                })
            
            # Extract related topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'duckduckgo_related'
                    })
            
            # If no results from instant API, try scraping search results
            if not results:
                results = self._scrape_duckduckgo_results(query, max_results)
            
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error searching DuckDuckGo: {e}")
            return []
    
    def _scrape_duckduckgo_results(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Scrape DuckDuckGo search results as fallback.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of search result dictionaries
        """
        try:
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Find search result elements
            result_elements = soup.find_all('div', class_='result')
            
            for element in result_elements[:max_results]:
                try:
                    title_elem = element.find('a', class_='result__a')
                    snippet_elem = element.find('a', class_='result__snippet')
                    
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        url = title_elem.get('href', '')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'source': 'duckduckgo_scrape'
                        })
                        
                except Exception as e:
                    self.logger.debug(f"Error parsing search result element: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error scraping DuckDuckGo results: {e}")
            return []
    
    def _process_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Process and combine search results into coherent content.
        
        Args:
            search_results: List of search result dictionaries
            
        Returns:
            Processed content string
        """
        if not search_results:
            return ""
        
        processed_parts = []
        
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'Untitled')
            snippet = result.get('snippet', '')
            url = result.get('url', '')
            
            if snippet:
                # Clean up the snippet
                cleaned_snippet = self._clean_text(snippet)
                
                # Format the result
                result_text = f"**Resultat {i}: {title}**\n{cleaned_snippet}"
                if url:
                    result_text += f"\nKilde: {url}"
                
                processed_parts.append(result_text)
        
        return "\n\n".join(processed_parts)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        
        # Remove URLs from text content
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def _calculate_search_confidence(self, query: str, search_results: List[Dict], processed_content: str) -> float:
        """
        Calculate confidence score for search results.
        
        Args:
            query: Original query
            search_results: List of search results
            processed_content: Processed content string
            
        Returns:
            Confidence score between 0 and 1
        """
        if not search_results or not processed_content:
            return 0.0
        
        # Base confidence on number of results
        result_score = min(len(search_results) / 5.0, 1.0)
        
        # Check for priority domain results
        priority_score = 0.0
        for result in search_results:
            url = result.get('url', '').lower()
            if any(domain in url for domain in self.priority_domains):
                priority_score += 0.2
        priority_score = min(priority_score, 1.0)
        
        # Check query term coverage in results
        query_words = set(query.lower().split())
        content_words = set(processed_content.lower().split())
        coverage = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
        
        # Combine scores
        confidence = (result_score * 0.4) + (priority_score * 0.3) + (coverage * 0.3)
        
        return min(confidence, 1.0)
    
    def search_specific_domain(self, query: str, domain: str) -> Optional[AgentResponse]:
        """
        Search within a specific domain.
        
        Args:
            query: Search query
            domain: Domain to search within
            
        Returns:
            AgentResponse with domain-specific results
        """
        try:
            domain_query = f"site:{domain} {query}"
            return self.search_web(domain_query, max_results=3)
            
        except Exception as e:
            self.logger.error(f"Error searching domain {domain}: {e}")
            return None

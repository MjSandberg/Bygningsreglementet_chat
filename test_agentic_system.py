"""Test script for the agentic RAG system."""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.agents import AgenticRAGSystem
from src.scraper import WebScraper
from src.utils.logging import setup_logging


def test_agentic_system():
    """Test the agentic RAG system."""
    # Setup logging
    setup_logging(level="INFO")
    
    print("🚀 Testing Agentic RAG System")
    print("=" * 50)
    
    try:
        # Initialize data
        print("📊 Loading data...")
        scraper = WebScraper()
        if not scraper.data:
            print("No local data found. This test requires scraped data.")
            print("Please run the scraper first or use a smaller test dataset.")
            # Create a small test dataset
            test_data = [
                "Bygningsreglementet - Administrative bestemmelser: Dette er en test af administrative bestemmelser for bygninger.",
                "Section 02 - Brandkrav: Bygninger skal opfylde brandkrav for at sikre sikkerheden.",
                "Section 03 - Konstruktion: Konstruktive elementer skal dimensioneres korrekt.",
                "Section 04 - Installationer: Tekniske installationer skal overholde gældende standarder.",
                "Bilag 1 - Definitioner: Her findes definitioner af vigtige begreber i bygningsreglementet."
            ]
            data = test_data
        else:
            data = scraper.get_data()[:100]  # Use first 100 items for testing
        
        print(f"✅ Loaded {len(data)} data items")
        
        # Initialize agentic system
        print("🤖 Initializing agentic system...")
        agentic_system = AgenticRAGSystem(data)
        print("✅ Agentic system initialized")
        
        # Test system status
        print("\n🔍 System Status:")
        status = agentic_system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test queries
        test_queries = [
            "Hvad er brandkrav for bygninger?",
            "Hvordan dimensioneres konstruktive elementer?",
            "Hvad betyder BR18?",
            "Seneste opdateringer til bygningsreglementet"
        ]
        
        print("\n🧪 Testing Queries:")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            try:
                response = agentic_system.process_query(query)
                print(f"   Response length: {len(response)} characters")
                print(f"   Response preview: {response[:200]}...")
                print("   ✅ Success")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test agent statistics
        print("\n📈 Agent Statistics:")
        stats = agentic_system.get_agent_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test individual agents
        print("\n🔧 Testing Individual Agents:")
        test_results = agentic_system.test_agents()
        overall_success_rate = test_results.pop('overall_success_rate', 0)
        
        for agent, result in test_results.items():
            if isinstance(result, dict):
                status = "✅" if result.get("success", False) else "❌"
                print(f"  {agent}: {status} {result}")
            else:
                print(f"  {agent}: {result}")
        
        print(f"\n🎉 Overall Success Rate: {overall_success_rate:.2%}")
        
        # Test citation functionality
        print("\n📚 Testing Citation Functionality:")
        try:
            test_query = "Hvad er brandkrav for bygninger?"
            response = agentic_system.process_query(test_query)
            citation_mapping = agentic_system.get_last_citation_mapping()
            
            if citation_mapping:
                print(f"   ✅ Citations generated: {len(citation_mapping.citations)} sources")
                formatted_citations = agentic_system.citation_agent.format_citations_for_display(citation_mapping)
                print(f"   ✅ Citations formatted for UI: {formatted_citations['total_citations']} citations")
                
                # Test citation validation
                validation = agentic_system.citation_agent.validate_citations(citation_mapping)
                print(f"   ✅ Citation quality score: {validation.get('quality_score', 0):.2f}")
            else:
                print("   ⚠️  No citations generated")
                
        except Exception as e:
            print(f"   ❌ Citation test failed: {e}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_agentic_system()

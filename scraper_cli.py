"""CLI script for scraping Bygningsreglementet data."""

import sys
import argparse
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.utils.logging import setup_logging
from src.scraper import WebScraper


def main():
    """Main function for the scraper CLI."""
    parser = argparse.ArgumentParser(description='Scrape Bygningsreglementet')
    parser.add_argument(
        '--force', 
        action='store_true', 
        help='Force rescraping even if data exists'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    
    # Run scraper
    scraper = WebScraper(force_rescrape=args.force)
    scraper.scrape_all()
    
    print(f"Scraping completed. {len(scraper.get_data())} passages collected.")


if __name__ == "__main__":
    main()

"""Web scraper for Bygningsreglementet content."""

from typing import List, Optional
import requests
import json
import os
from bs4 import BeautifulSoup
from pathlib import Path

from ..config import Config
from ..utils.logging import get_logger
from ..utils.text_processing import fix_text, split_text_if_needed


class WebScraper:
    """Web scraper for Danish Building Regulations content."""
    
    def __init__(self, force_rescrape: bool = False):
        """
        Initialize the web scraper.
        
        Args:
            force_rescrape: Whether to force rescraping even if data exists
        """
        self.logger = get_logger("scraper")
        self.data_file = Config.get_data_file_path()
        self.force_rescrape = force_rescrape
        self.data: List[str] = []
        
        if not force_rescrape:
            self.load_data()

    def load_data(self) -> bool:
        """
        Load data from local storage if it exists.
        
        Returns:
            True if data was loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                self.logger.info(f"Loaded {len(self.data)} passages from {self.data_file}")
                return True
            else:
                self.logger.info("No existing data file found")
                return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Error loading data file: {e}")
            self.logger.info("Will start fresh scraping")
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error loading data: {e}")
            return False

    def save_data(self, data: List[str]) -> None:
        """
        Save data to local storage.
        
        Args:
            data: List of text passages to save
        """
        try:
            # Ensure data directory exists
            Path(self.data_file).parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving {len(data)} passages to {self.data_file}")
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info("Save completed successfully")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise

    def fetch_content(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch content from URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            self.logger.debug(f"Fetching content from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None

    def process_tekniske_bestemmelser(self, soup: BeautifulSoup, section_name: str) -> List[str]:
        """
        Process content from pages.
        
        Args:
            soup: BeautifulSoup object containing the page content
            section_name: Name of the section being processed
            
        Returns:
            List of processed text passages
        """
        processed_data = []
        if not soup:
            return processed_data

        accordion_rows = soup.find_all('div', class_='accordion')
        for row in accordion_rows:
            content = row.find('div', class_='accordion__content')
            if content:
                content_text = fix_text(
                    content.get_text(strip=True).replace('\xa0', ' ')
                )
                head = row.find("div", class_="accordion__header")
                if head:
                    header = f"{section_name} - {fix_text(head.get_text(strip=True))}"
                    processed_data.extend(
                        split_text_if_needed(content_text, header)
                    )
        
        return processed_data

    def scrape_administrative_bestemmelser(self) -> List[str]:
        """
        Scrape Administrative-bestemmelser section.
        
        Returns:
            List of processed text passages
        """
        url = "https://bygningsreglementet.dk/Administrative-bestemmelser/Krav?Layout=ShowAll"
        self.logger.info("Processing Administrative-bestemmelser")
        
        soup = self.fetch_content(url)
        if soup:
            data = self.process_tekniske_bestemmelser(soup, "Administrative")
            self.logger.info(f"Found {len(data)} passages in Administrative-bestemmelser")
            return data
        return []

    def scrape_tekniske_bestemmelser(self) -> List[str]:
        """
        Scrape Tekniske-bestemmelser sections (02-22).
        
        Returns:
            List of processed text passages
        """
        all_data = []
        self.logger.info("Scraping Tekniske-bestemmelser...")
        
        for section in range(2, 23):
            section_num = f"{section:02d}"  # Format as 02, 03, etc.
            url = f"https://bygningsreglementet.dk/Tekniske-bestemmelser/{section_num}/Krav?Layout=ShowAll"
            self.logger.info(f"Processing section {section_num}/22")
            
            soup = self.fetch_content(url)
            if soup:
                section_data = self.process_tekniske_bestemmelser(soup, f"Section {section_num}")
                self.logger.info(f"Found {len(section_data)} passages in section {section_num}")
                all_data.extend(section_data)
        
        return all_data

    def scrape_bilag(self) -> List[str]:
        """
        Scrape Bilag sections.
        
        Returns:
            List of processed text passages
        """
        all_data = []
        self.logger.info("Scraping Bilag...")
        
        for bilag in range(1, 7):
            url = f"https://bygningsreglementet.dk/Bilag/B{bilag}/Bilag_{bilag}"
            self.logger.info(f"Processing bilag {bilag}/6")
            
            soup = self.fetch_content(url)
            if soup:
                bilag_data = self.process_tekniske_bestemmelser(soup, f"Bilag {bilag}")
                self.logger.info(f"Found {len(bilag_data)} passages in bilag {bilag}")
                all_data.extend(bilag_data)
        
        return all_data

    def scrape_all(self) -> None:
        """Scrape all relevant URLs and store the data."""
        # If we have data and aren't forcing a rescrape, return early
        if self.data and not self.force_rescrape:
            self.logger.info(f"Using existing data with {len(self.data)} passages")
            return

        all_data = []
        
        # Scrape all sections
        all_data.extend(self.scrape_administrative_bestemmelser())
        all_data.extend(self.scrape_tekniske_bestemmelser())
        all_data.extend(self.scrape_bilag())

        # Save all collected data at once
        self.logger.info(f"Total passages collected: {len(all_data)}")
        self.save_data(all_data)
        self.data = all_data

    def get_data(self) -> List[str]:
        """
        Get the current data.
        
        Returns:
            List of text passages
        """
        return self.data

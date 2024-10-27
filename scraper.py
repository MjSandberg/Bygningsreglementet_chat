from typing import List, Dict
import requests
import re
import json
import os
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

class WebScraper:
    def __init__(self, force_rescrape=False):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.data_file = "bygningsreglementet_data.json"
        self.force_rescrape = force_rescrape
        self.data = []
        if not force_rescrape:
            self.load_data()

    def load_data(self) -> bool:
        """Load data from local storage if it exists"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                print(f"Loaded {len(self.data)} passages from {self.data_file}")
                return
            else:
                return False
        except json.JSONDecodeError as e:
            print(f"Error loading data file: {e}")
            print("Will start fresh scraping")
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            return False
        except Exception as e:
            print(f"Unexpected error loading data: {e}")
            return False

    def save_data(self, data: List[str]):
        """Save data to local storage"""
        try:
            print(f"Saving {len(data)} passages to {self.data_file}")
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("Save completed successfully")
        except Exception as e:
            print(f"Error saving data: {e}")

    def fix_text(self, text: str) -> str:
        """Fix text formatting issues"""
        # Fix punctuation
        text = re.sub(r'\.(\S)', r'. \1', text)
        
        # Add space between numbers and letters
        text = re.sub(r'(\d)([a-zA-ZæøåÆØÅ])', r'\1 \2', text)
        
        # Add space between letters and numbers
        text = re.sub(r'([a-zA-ZæøåÆØÅ])(\d)', r'\1 \2', text)
        
        # Fix spaces around section symbols
        text = re.sub(r'§(\S)', r'§ \1', text)
        text = re.sub(r'(\S)§', r'\1 §', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def fetch_content(self, url: str) -> BeautifulSoup:
        """Fetch content from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def process_tekniske_bestemmelser(self, soup: BeautifulSoup, section_name: str) -> List[str]:
        """Process content from pages"""
        processed_data = []
        if not soup:
            return processed_data

        accordion_rows = soup.find_all('div', class_='accordion')
        for row in accordion_rows:
            content = row.find('div', class_='accordion__content')
            if content:
                content = self.fix_text(content.get_text(strip=True).replace(u'\xa0', u' '))
                head = row.find("div", class_="accordion__header")
                if head:
                    head = f"{section_name} - {self.fix_text(head.get_text(strip=True))}"
                    if len(content) > 250:
                        content_list = self.text_splitter.split_text(content)
                        for x in content_list:
                            processed_data.append(head + ": " + x)
                    else:
                        processed_data.append(head + ": " + content)
        return processed_data

    def scrape_all(self):
        """Scrape all relevant URLs and store the data"""
        # If we have data and aren't forcing a rescrape, return early
        if self.data and not self.force_rescrape:
            print(f"Using existing data with {len(self.data)} passages")
            return

        all_data = []
        
        # Scrape Administrative-bestemmelser
        admin_url = "https://bygningsreglementet.dk/Administrative-bestemmelser/Krav?Layout=ShowAll"
        print(f"Processing Administrative-bestemmelser")
        soup = self.fetch_content(admin_url)
        if soup:
            admin_data = self.process_tekniske_bestemmelser(soup, "Administrative")
            print(f"Found {len(admin_data)} passages in Administrative-bestemmelser")
            all_data.extend(admin_data)

        # Scrape Tekniske-bestemmelser (02-22)
        print("\nScraping Tekniske-bestemmelser...")
        for section in range(2, 23):
            section_num = f"{section:02d}"  # Format as 02, 03, etc.
            url = f"https://bygningsreglementet.dk/Tekniske-bestemmelser/{section_num}/Krav?Layout=ShowAll"
            print(f"Processing section {section_num}/22")
            soup = self.fetch_content(url)
            if soup:
                section_data = self.process_tekniske_bestemmelser(soup, f"Section {section_num}")
                print(f"Found {len(section_data)} passages in section {section_num}")
                all_data.extend(section_data)

        # Scrape Bilag
        print("\nScraping Bilag...")
        for bilag in range(1, 7):
            url = f"https://bygningsreglementet.dk/Bilag/B{bilag}/Bilag_{bilag}"
            print(f"Processing bilag {bilag}/6")
            soup = self.fetch_content(url)
            if soup:
                bilag_data = self.process_tekniske_bestemmelser(soup, bilag)
                print(f"Found {len(bilag_data)} passages in bilag {bilag}")
                all_data.extend(bilag_data)

        # Save all collected data at once
        print(f"\nTotal passages collected: {len(all_data)}")
        self.save_data(all_data)
        self.data = all_data

    def get_data(self) -> List[str]:
        """Get the current data"""
        return self.data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape Bygningsreglementet')
    parser.add_argument('--force', action='store_true', help='Force rescraping even if data exists')
    args = parser.parse_args()
    
    scraper = WebScraper(force_rescrape=args.force)
    scraper.scrape_all()

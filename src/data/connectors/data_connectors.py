"""
Data connectors for various data sources.
"""
import aiohttp
import asyncio
import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import os
from tqdm import tqdm
import json
import requests
from pathlib import Path

class DataConnector:
    """Base class for data connectors."""
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def fetch(self) -> List[Dict[str, Any]]:
        """Fetch data from the source."""
        raise NotImplementedError
        
    def save(self, data: List[Dict[str, Any]], filename: str):
        """Save data to file."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(data)} items to {output_path}")

class GutenbergConnector(DataConnector):
    """Connector for Project Gutenberg."""
    def __init__(self, output_dir: str, max_books: int = 100):
        super().__init__(output_dir)
        self.max_books = max_books
        self.base_url = "https://www.gutenberg.org"
        
    async def fetch(self) -> List[Dict[str, Any]]:
        books = []
        async with aiohttp.ClientSession() as session:
            # Get list of recent books
            async with session.get(f"{self.base_url}/browse/recent/") as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    book_links = soup.find_all('a', href=lambda x: x and x.startswith('/ebooks/'))
                    
                    for link in tqdm(book_links[:self.max_books], desc="Fetching books"):
                        book_id = link['href'].split('/')[-1]
                        book_url = f"{self.base_url}/files/{book_id}/{book_id}-h/{book_id}-h.htm"
                        
                        try:
                            async with session.get(book_url) as book_response:
                                if book_response.status == 200:
                                    book_html = await book_response.text()
                                    book_soup = BeautifulSoup(book_html, 'html.parser')
                                    text = book_soup.get_text()
                                    
                                    books.append({
                                        'id': book_id,
                                        'title': link.text.strip(),
                                        'text': text
                                    })
                        except Exception as e:
                            logging.error(f"Error fetching book {book_id}: {e}")
                            
        return books

class WikipediaConnector(DataConnector):
    """Connector for Wikipedia dumps."""
    def __init__(self, output_dir: str, language: str = 'en'):
        super().__init__(output_dir)
        self.language = language
        self.dump_url = f"https://dumps.wikimedia.org/{language}wiki/latest/{language}wiki-latest-pages-articles.xml.bz2"
        
    async def fetch(self) -> List[Dict[str, Any]]:
        # Download the dump file
        dump_path = self.output_dir / f"{self.language}wiki-latest-pages-articles.xml.bz2"
        
        if not dump_path.exists():
            response = requests.get(self.dump_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dump_path, 'wb') as f, tqdm(
                desc="Downloading Wikipedia dump",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
        
        # Process the dump file (simplified version)
        # In a real implementation, you would use a proper XML parser
        # and process the dump file in chunks
        return []

class ArxivConnector(DataConnector):
    """Connector for arXiv papers."""
    def __init__(self, output_dir: str, max_papers: int = 100):
        super().__init__(output_dir)
        self.max_papers = max_papers
        self.base_url = "http://export.arxiv.org/api/query"
        
    async def fetch(self) -> List[Dict[str, Any]]:
        papers = []
        search_query = 'cat:cs.AI OR cat:cs.CL OR cat:cs.LG'
        
        async with aiohttp.ClientSession() as session:
            params = {
                'search_query': search_query,
                'max_results': self.max_papers,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    xml = await response.text()
                    soup = BeautifulSoup(xml, 'xml')
                    
                    for entry in soup.find_all('entry'):
                        paper = {
                            'id': entry.id.text.split('/')[-1],
                            'title': entry.title.text,
                            'abstract': entry.summary.text,
                            'authors': [author.name.text for author in entry.find_all('author')],
                            'published': entry.published.text
                        }
                        papers.append(paper)
                        
        return papers

class DataConnectorFactory:
    """Factory for creating data connectors."""
    @staticmethod
    def create_connector(source: str, output_dir: str, **kwargs) -> DataConnector:
        connectors = {
            'gutenberg': GutenbergConnector,
            'wikipedia': WikipediaConnector,
            'arxiv': ArxivConnector
        }
        
        if source not in connectors:
            raise ValueError(f"Unknown data source: {source}")
            
        return connectors[source](output_dir, **kwargs) 
import requests
from bs4 import BeautifulSoup
import json
import os
import time
from urllib.parse import urljoin

class WebScraper:
    def __init__(self, base_url):
        """
        Initialize the web scraper with the base URL of the website.
        
        Args:
            base_url (str): Base URL of the website to scrape
        """
        self.base_url = base_url
        self.visited_urls = set()
        self.content = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_page_content(self, url):
        """
        Get the content of a page.
        
        Args:
            url (str): URL of the page to scrape
            
        Returns:
            tuple: (soup, text) where soup is the BeautifulSoup object and text is the extracted text
        """
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from the page
            text = ""
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text += tag.get_text() + "\n"
                
            return soup, text.strip()
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None, ""
    
    def extract_links(self, soup, current_url):
        """
        Extract links from a page.
        
        Args:
            soup (BeautifulSoup): BeautifulSoup object of the page
            current_url (str): URL of the current page
            
        Returns:
            list: List of extracted links
        """
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            
            # Skip empty links, anchors, and external links
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
                
            # Convert relative URLs to absolute URLs
            if not href.startswith('http'):
                href = urljoin(current_url, href)
                
            # Only include links from the same domain
            if self.base_url in href and href not in self.visited_urls:
                links.append(href)
                
        return links
    
    def scrape_page(self, url, depth=0, max_depth=2):
        """
        Scrape a page and its links recursively.
        
        Args:
            url (str): URL of the page to scrape
            depth (int): Current depth of recursion
            max_depth (int): Maximum depth of recursion
            
        Returns:
            dict: Dictionary containing the scraped content
        """
        if depth > max_depth or url in self.visited_urls:
            return
            
        print(f"Scraping {url} (depth: {depth})")
        self.visited_urls.add(url)
        
        soup, text = self.get_page_content(url)
        if not soup:
            return
            
        # Store the content
        page_title = soup.title.string if soup.title else url
        self.content[url] = {
            'title': page_title,
            'text': text
        }
        
        # Extract and follow links
        links = self.extract_links(soup, url)
        for link in links:
            time.sleep(1)  # Be nice to the server
            self.scrape_page(link, depth + 1, max_depth)
    
    def scrape_specific_pages(self, urls):
        """
        Scrape specific pages without recursion.
        
        Args:
            urls (list): List of URLs to scrape
            
        Returns:
            dict: Dictionary containing the scraped content
        """
        for url in urls:
            if url not in self.visited_urls:
                print(f"Scraping {url}")
                self.visited_urls.add(url)
                
                soup, text = self.get_page_content(url)
                if not soup:
                    continue
                    
                # Store the content
                page_title = soup.title.string if soup.title else url
                self.content[url] = {
                    'title': page_title,
                    'text': text
                }
                
                time.sleep(1)  # Be nice to the server
    
    def save_content(self, output_file):
        """
        Save the scraped content to a JSON file.
        
        Args:
            output_file (str): Path to the output JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.content, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving content to file: {e}")
            return False
    
    def process_content_for_rag(self, chunk_size=1000, chunk_overlap=200):
        """
        Process the scraped content into chunks suitable for RAG.
        
        Args:
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Overlap between chunks
            
        Returns:
            list: List of processed chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        # Combine all text from all pages
        all_text = ""
        for url, data in self.content.items():
            all_text += f"Title: {data['title']}\n\n"
            all_text += f"{data['text']}\n\n"
            all_text += f"Source: {url}\n\n"
            all_text += "---\n\n"
            
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(all_text)
        return chunks
    
    def save_chunks(self, chunks, output_file):
        """
        Save the processed chunks to a JSON file.
        
        Args:
            chunks (list): List of processed chunks
            output_file (str): Path to the output JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(chunks, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving chunks to file: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize the scraper
    scraper = WebScraper("https://www.leechy.app")
    
    # Define specific pages to scrape
    pages_to_scrape = [
        "https://www.leechy.app/",
        "https://www.leechy.app/terms-of-service",
        "https://www.leechy.app/privacy-policy"
    ]
    
    # Scrape the specific pages
    scraper.scrape_specific_pages(pages_to_scrape)
    
    # Save the raw content
    scraper.save_content("/home/ubuntu/leechy_chatbot/website_content.json")
    
    # Process the content for RAG
    chunks = scraper.process_content_for_rag()
    
    # Save the chunks
    scraper.save_chunks(chunks, "/home/ubuntu/leechy_chatbot/website_chunks.json")
    
    print(f"Scraped {len(scraper.content)} pages")
    print(f"Generated {len(chunks)} chunks for RAG")

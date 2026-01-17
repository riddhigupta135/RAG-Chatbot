"""
Web scraping utilities for extracting content from internal websites.
Handles HTML parsing, link extraction, and content cleaning.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedDocument:
    """Represents a scraped document with metadata."""
    url: str
    title: str
    content: str
    metadata: dict


class WebScraper:
    """
    Async web scraper for extracting content from internal documentation sites.
    
    Features:
    - Async HTTP requests for efficient scraping
    - HTML cleaning and text extraction
    - Metadata extraction (title, URL, etc.)
    - Rate limiting to avoid overloading servers
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        max_pages: int = 100,
        request_delay: float = 0.5,
    ):
        """
        Initialize the web scraper.
        
        Args:
            base_url: Base URL to restrict scraping scope.
            max_pages: Maximum number of pages to scrape.
            request_delay: Delay between requests in seconds.
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.request_delay = request_delay
        self.visited_urls: set[str] = set()
        
    async def scrape_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[ScrapedDocument]:
        """
        Scrape a single URL and extract content.
        
        Args:
            session: aiohttp client session.
            url: URL to scrape.
            
        Returns:
            ScrapedDocument if successful, None otherwise.
        """
        try:
            logger.info("scraping_url", url=url)
            
            async with session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(
                        "scrape_failed",
                        url=url,
                        status=response.status,
                    )
                    return None
                    
                html = await response.text()
                
            # Parse HTML
            soup = BeautifulSoup(html, "lxml")
            
            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract main content
            main_content = soup.find("main") or soup.find("article") or soup.body
            
            if not main_content:
                logger.warning("no_content_found", url=url)
                return None
                
            # Clean text
            text = main_content.get_text(separator="\n", strip=True)
            
            # Remove excessive whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)
            
            if len(content) < 50:  # Skip pages with minimal content
                logger.debug("content_too_short", url=url, length=len(content))
                return None
                
            logger.info(
                "scrape_success",
                url=url,
                title=title,
                content_length=len(content),
            )
            
            return ScrapedDocument(
                url=url,
                title=str(title).strip(),
                content=content,
                metadata={
                    "source": url,
                    "title": str(title).strip(),
                    "type": "webpage",
                },
            )
            
        except Exception as e:
            logger.error("scrape_error", url=url, error=str(e))
            return None
            
    def extract_links(self, html: str, base_url: str) -> list[str]:
        """
        Extract internal links from HTML content.
        
        Args:
            html: HTML content to parse.
            base_url: Base URL for resolving relative links.
            
        Returns:
            List of absolute URLs found in the page.
        """
        soup = BeautifulSoup(html, "lxml")
        links = []
        
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            absolute_url = urljoin(base_url, href)
            
            # Only include internal links
            if self.base_url and absolute_url.startswith(self.base_url):
                # Remove fragments
                parsed = urlparse(absolute_url)
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                links.append(clean_url)
                
        return list(set(links))
        
    async def scrape_site(
        self,
        start_url: str,
        follow_links: bool = True,
    ) -> list[ScrapedDocument]:
        """
        Scrape an entire site starting from a URL.
        
        Args:
            start_url: Starting URL for the crawl.
            follow_links: Whether to follow internal links.
            
        Returns:
            List of scraped documents.
        """
        documents = []
        urls_to_visit = [start_url]
        
        if not self.base_url:
            parsed = urlparse(start_url)
            self.base_url = f"{parsed.scheme}://{parsed.netloc}"
            
        async with aiohttp.ClientSession() as session:
            while urls_to_visit and len(documents) < self.max_pages:
                url = urls_to_visit.pop(0)
                
                if url in self.visited_urls:
                    continue
                    
                self.visited_urls.add(url)
                
                doc = await self.scrape_url(session, url)
                
                if doc:
                    documents.append(doc)
                    
                    if follow_links:
                        # Fetch HTML again to extract links
                        try:
                            async with session.get(url) as response:
                                html = await response.text()
                                new_links = self.extract_links(html, url)
                                urls_to_visit.extend(
                                    link for link in new_links
                                    if link not in self.visited_urls
                                )
                        except Exception:
                            pass
                            
                # Rate limiting
                await asyncio.sleep(self.request_delay)
                
        logger.info(
            "scrape_complete",
            total_documents=len(documents),
            pages_visited=len(self.visited_urls),
        )
        
        return documents

"""
Forum scraping functionality for Forum Wisdom Miner.

This module handles fetching and parsing forum threads from various forum platforms.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import (
    DELAY_BETWEEN_REQUESTS, POST_SELECTORS, 
    AUTHOR_SELECTORS, DATE_SELECTORS
)
from utils.helpers import normalize_url, post_hash
from utils.text_utils import clean_post_content

logger = logging.getLogger(__name__)


class ForumScraper:
    """Enhanced forum scraper with robust error handling and retry logic."""
    
    def __init__(self, delay: float = None):
        """Initialize the forum scraper.
        
        Args:
            delay: Delay between requests in seconds
        """
        self.delay = delay or DELAY_BETWEEN_REQUESTS
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_posts_scraped': 0,
            'start_time': time.time()
        }
    
    def scrape_thread(self, base_url: str, max_pages: int = 50) -> Tuple[List[Dict], Dict]:
        """Scrape an entire forum thread across multiple pages.
        
        Args:
            base_url: Base URL of the thread
            max_pages: Maximum number of pages to scrape
            
        Returns:
            Tuple of (posts_list, metadata_dict)
        """
        logger.info(f"Starting thread scrape: {base_url}")
        self.stats['start_time'] = time.time()
        
        normalized_url = normalize_url(base_url)
        all_posts = []
        metadata = {
            'base_url': normalized_url,
            'pages_scraped': 0,
            'total_posts': 0,
            'scrape_timestamp': time.time(),
            'errors': []
        }
        
        current_url = normalized_url
        page_num = 1
        global_post_position = 1
        
        while current_url and page_num <= max_pages:
            try:
                logger.info(f"Scraping page {page_num}: {current_url}")
                
                page_posts, next_url = self._scrape_single_page(
                    current_url, page_num, global_post_position
                )
                
                if page_posts:
                    all_posts.extend(page_posts)
                    global_post_position += len(page_posts)
                    metadata['pages_scraped'] += 1
                    logger.info(f"Page {page_num}: Found {len(page_posts)} posts")
                else:
                    logger.warning(f"Page {page_num}: No posts found")
                
                # Check for next page
                if next_url and next_url != current_url:
                    current_url = next_url
                    page_num += 1
                else:
                    logger.info("No more pages found or reached end of thread")
                    break
                
                # Rate limiting
                if self.delay > 0:
                    time.sleep(self.delay)
                    
            except Exception as e:
                error_msg = f"Error scraping page {page_num}: {e}"
                logger.error(error_msg)
                metadata['errors'].append(error_msg)
                self.stats['failed_requests'] += 1
                break
        
        metadata['total_posts'] = len(all_posts)
        metadata['scrape_duration'] = time.time() - self.stats['start_time']
        self.stats['total_posts_scraped'] += len(all_posts)
        
        logger.info(f"Thread scrape complete: {len(all_posts)} posts from {metadata['pages_scraped']} pages")
        return all_posts, metadata
    
    def _scrape_single_page(self, url: str, page_num: int, start_position: int) -> Tuple[List[Dict], Optional[str]]:
        """Scrape a single page of a forum thread.
        
        Args:
            url: URL of the page to scrape
            page_num: Page number for reference
            start_position: Starting global post position
            
        Returns:
            Tuple of (posts_list, next_page_url)
        """
        try:
            response = self._fetch_page(url)
            if not response:
                return [], None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            posts = self._extract_posts(soup, page_num, start_position)
            next_url = self._find_next_page_url(soup, url)
            
            return posts, next_url
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return [], None
    
    def _fetch_page(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Fetch a page with retry logic.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            Response object or None if failed
        """
        for attempt in range(max_retries):
            try:
                self.stats['total_requests'] += 1
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                self.stats['successful_requests'] += 1
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.stats['failed_requests'] += 1
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
        
        return None
    
    def _extract_posts(self, soup: BeautifulSoup, page_num: int, start_position: int) -> List[Dict]:
        """Extract posts from a page's HTML.
        
        Args:
            soup: BeautifulSoup object of the page
            page_num: Current page number
            start_position: Starting global position for posts
            
        Returns:
            List of post dictionaries
        """
        posts = []
        
        # Try different post selectors
        post_elements = None
        for selector in POST_SELECTORS:
            post_elements = soup.select(selector)
            if post_elements:
                logger.debug(f"Found {len(post_elements)} posts using selector: {selector}")
                break
        
        if not post_elements:
            logger.warning("No posts found with any known selector")
            return posts
        
        for i, post_elem in enumerate(post_elements):
            try:
                post_data = self._extract_single_post(
                    post_elem, page_num, start_position + i
                )
                if post_data:
                    posts.append(post_data)
                    
            except Exception as e:
                logger.warning(f"Error extracting post {i} on page {page_num}: {e}")
                continue
        
        return posts
    
    def _extract_single_post(self, post_elem, page_num: int, global_position: int) -> Optional[Dict]:
        """Extract data from a single post element.
        
        Args:
            post_elem: BeautifulSoup element containing the post
            page_num: Current page number
            global_position: Global position of this post
            
        Returns:
            Post dictionary or None if extraction failed
        """
        try:
            # Extract content
            content_raw = str(post_elem)
            content_clean = clean_post_content(content_raw)
            
            if not content_clean or len(content_clean.strip()) < 10:
                return None
            
            # Extract author
            author = self._extract_author(post_elem)
            
            # Extract date
            date = self._extract_date(post_elem)
            
            # Create post data
            post_data = {
                'content': content_clean,
                'author': author,
                'date': date,
                'page': page_num,
                'position_on_page': global_position - ((page_num - 1) * 20),  # Approximate
                'global_position': global_position,
                'url': '',  # Will be set by caller
                'hash': post_hash(content_clean, author, date)
            }
            
            return post_data
            
        except Exception as e:
            logger.warning(f"Error extracting post data: {e}")
            return None
    
    def _extract_author(self, post_elem) -> str:
        """Extract author from post element."""
        for selector in AUTHOR_SELECTORS:
            author_elem = post_elem.select_one(selector)
            if author_elem:
                author = author_elem.get_text(strip=True)
                if author and len(author) < 100:  # Sanity check
                    return author
        
        return 'unknown-author'
    
    def _extract_date(self, post_elem) -> str:
        """Extract date from post element."""
        for selector in DATE_SELECTORS:
            date_elem = post_elem.select_one(selector)
            if date_elem:
                # Try text content first
                date_text = date_elem.get_text(strip=True)
                if date_text and len(date_text) < 100:
                    return date_text
                
                # Try common date attributes
                for attr in ['datetime', 'title', 'data-time']:
                    date_attr = date_elem.get(attr)
                    if date_attr:
                        return str(date_attr)
        
        return 'unknown-date'
    
    def _find_next_page_url(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find the URL of the next page.
        
        Args:
            soup: BeautifulSoup object of current page
            current_url: URL of current page
            
        Returns:
            URL of next page or None
        """
        # Common patterns for next page links
        next_selectors = [
            'a[rel="next"]',
            'a:contains("Next")',
            'a:contains(">")',
            '.pagination a:contains("Next")',
            '.pagelinks a:contains("Next")',
            'a[title*="Next"]'
        ]
        
        for selector in next_selectors:
            try:
                next_elem = soup.select_one(selector)
                if next_elem and next_elem.get('href'):
                    next_url = urljoin(current_url, next_elem['href'])
                    if next_url != current_url:
                        return next_url
            except Exception:
                continue
        
        # Try to find numbered pagination
        try:
            current_page_num = self._extract_current_page_number(soup)
            if current_page_num:
                next_page_link = soup.select_one(f'a:contains("{current_page_num + 1}")')
                if next_page_link and next_page_link.get('href'):
                    return urljoin(current_url, next_page_link['href'])
        except Exception:
            pass
        
        return None
    
    def _extract_current_page_number(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract current page number from pagination."""
        # Try to find current page in pagination
        for selector in ['.pagination .current', '.pagelinks .current', '.page-current']:
            current_elem = soup.select_one(selector)
            if current_elem:
                try:
                    return int(current_elem.get_text(strip=True))
                except ValueError:
                    continue
        return None
    
    def get_stats(self) -> Dict:
        """Get scraping statistics."""
        duration = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'duration': duration,
            'success_rate': (
                self.stats['successful_requests'] / max(1, self.stats['total_requests'])
            ),
            'posts_per_second': self.stats['total_posts_scraped'] / max(1, duration)
        }


__all__ = ['ForumScraper']
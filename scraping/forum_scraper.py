"""
Forum scraping functionality for Forum Wisdom Miner.

This module handles fetching and parsing forum threads from various forum platforms.
"""

import logging
import os
import random
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config.settings import (
    BASE_REQUEST_DELAY, JITTER_RANGE, RESPECTFUL_SCRAPING, USER_AGENTS, 
    RETRY_CONFIG, DELAY_BETWEEN_REQUESTS
)
from config.platform_config import get_platform_config, detect_forum_platform
from utils.helpers import normalize_url, post_hash
from utils.text_utils import clean_post_content
from utils.date_parser import parse_forum_date
from utils.monitoring import monitor_scraping_operation

logger = logging.getLogger(__name__)


class ForumScraper:
    """Enhanced forum scraper with dynamic platform configuration support."""
    
    def __init__(self, delay: float = None, platform_config: Dict = None, respectful: bool = True):
        """Initialize the forum scraper.
        
        Args:
            delay: Delay between requests in seconds (overrides platform config)
            platform_config: Platform-specific configuration dict
            respectful: Enable respectful scraping with jitter and user-agent rotation
        """
        self.platform_config = platform_config or {}
        self.respectful = respectful and RESPECTFUL_SCRAPING
        
        # Set delay - use respectful delay by default, allow override
        if delay is not None:
            self.base_delay = delay
        elif self.respectful:
            self.base_delay = self.platform_config.get('scraping', {}).get('delay_between_requests', BASE_REQUEST_DELAY)
        else:
            self.base_delay = DELAY_BETWEEN_REQUESTS
        
        # Initialize session with rotating headers
        self.session = requests.Session()
        self._update_session_headers()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_posts_scraped': 0,
            'start_time': time.time(),
            'platform_detected': self.platform_config.get('platform', {}).get('name', 'Unknown'),
            'total_delay_time': 0,
            'respectful_mode': self.respectful
        }
    
    def _update_session_headers(self):
        """Update session headers with a random User-Agent if respectful mode is enabled."""
        if self.respectful:
            user_agent = random.choice(USER_AGENTS)
        else:
            user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        
        default_headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        platform_headers = self.platform_config.get('scraping', {}).get('headers', {})
        self.session.headers.update({**default_headers, **platform_headers})
        
        logger.debug(f"Updated User-Agent: {user_agent}")
    
    def _respectful_delay(self):
        """Apply respectful delay with jitter between requests."""
        if not self.respectful or self.base_delay <= 0:
            return
        
        # Add random jitter to avoid predictable patterns
        jitter = random.uniform(*JITTER_RANGE)
        total_delay = self.base_delay + jitter
        
        logger.debug(f"Respectful delay: {total_delay:.2f}s (base: {self.base_delay}s + jitter: {jitter:.2f}s)")
        
        start_time = time.time()
        time.sleep(total_delay)
        actual_delay = time.time() - start_time
        self.stats['total_delay_time'] += actual_delay
    
    @monitor_scraping_operation
    def scrape_thread(self, base_url: str, max_pages: int = 1000, save_html: bool = True, thread_dir: str = None) -> Tuple[List[Dict], Dict]:
        """Scrape an entire forum thread across multiple pages.
        
        Args:
            base_url: Base URL of the thread
            max_pages: Maximum number of pages to scrape
            save_html: Whether to save raw HTML files for reprocessing
            thread_dir: Directory to save HTML files (required if save_html=True)
            
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
            'errors': [],
            'html_files_saved': []
        }
        
        # Create HTML directory if saving HTML
        if save_html and thread_dir:
            html_dir = os.path.join(thread_dir, 'html_pages')
            os.makedirs(html_dir, exist_ok=True)
        
        current_url = normalized_url
        page_num = 1
        global_post_position = 1
        
        while current_url and page_num <= max_pages:
            try:
                logger.info(f"Scraping page {page_num}: {current_url}")
                
                # Determine HTML file path
                html_file_path = None
                if save_html and thread_dir:
                    html_file_path = os.path.join(html_dir, f"page_{page_num:03d}.html")
                
                page_posts, next_url = self._scrape_single_page(
                    current_url, page_num, global_post_position, html_file_path
                )
                
                if page_posts:
                    all_posts.extend(page_posts)
                    global_post_position += len(page_posts)
                    metadata['pages_scraped'] += 1
                    logger.info(f"Page {page_num}: Found {len(page_posts)} posts")
                    
                    # Track saved HTML file
                    if html_file_path and os.path.exists(html_file_path):
                        metadata['html_files_saved'].append(f"page_{page_num:03d}.html")
                else:
                    logger.warning(f"Page {page_num}: No posts found")
                
                # Check for next page
                if next_url and next_url != current_url:
                    current_url = next_url
                    page_num += 1
                else:
                    logger.info("No more pages found or reached end of thread")
                    break
                
                # Respectful rate limiting with jitter
                self._respectful_delay()
                
                # Rotate User-Agent occasionally to appear more natural
                if self.respectful and page_num % 5 == 0:
                    self._update_session_headers()
                    
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
    
    def _scrape_single_page(self, url: str, page_num: int, start_position: int, html_file_path: str = None) -> Tuple[List[Dict], Optional[str]]:
        """Scrape a single page of a forum thread.
        
        Args:
            url: URL of the page to scrape
            page_num: Page number for reference
            start_position: Starting global post position
            html_file_path: Optional path to save HTML content
            
        Returns:
            Tuple of (posts_list, next_page_url)
        """
        try:
            response = self._fetch_page(url)
            if not response:
                return [], None
            
            # Save HTML content if requested
            if html_file_path:
                try:
                    with open(html_file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    logger.debug(f"Saved HTML to {html_file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save HTML to {html_file_path}: {e}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            posts = self._extract_posts(soup, page_num, start_position, url)
            next_url = self._find_next_page_url(soup, url)
            
            return posts, next_url
            
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return [], None
    
    def _fetch_page(self, url: str, max_retries: int = None) -> Optional[requests.Response]:
        """Fetch a page with respectful retry logic and exponential backoff.
        
        Args:
            url: URL to fetch
            max_retries: Maximum number of retry attempts (uses config if None)
            
        Returns:
            Response object or None if failed
        """
        if max_retries is None:
            max_retries = RETRY_CONFIG['scraping']['max_retries']
        
        backoff_multiplier = RETRY_CONFIG['scraping']['backoff']
        base_retry_delay = RETRY_CONFIG['scraping'].get('base_delay', 2.0)
        
        for attempt in range(max_retries):
            try:
                self.stats['total_requests'] += 1
                
                # Add a small delay before each request to be respectful
                if attempt > 0 and self.respectful:
                    # Exponential backoff with jitter for retries
                    retry_delay = base_retry_delay * (backoff_multiplier ** (attempt - 1))
                    jitter = random.uniform(0.1, 0.5)
                    total_retry_delay = retry_delay + jitter
                    
                    logger.info(f"Retry delay for attempt {attempt + 1}: {total_retry_delay:.2f}s")
                    time.sleep(total_retry_delay)
                    self.stats['total_delay_time'] += total_retry_delay
                
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                self.stats['successful_requests'] += 1
                
                # Check if we're being rate limited or blocked
                if response.status_code == 429:
                    logger.warning(f"Rate limited by server: {url}")
                    raise requests.exceptions.RequestException("Rate limited")
                
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                self.stats['failed_requests'] += 1
                
                # If this is the last attempt, don't retry
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                    break
        
        return None
    
    def _extract_posts(self, soup: BeautifulSoup, page_num: int, start_position: int, page_url: str = None) -> List[Dict]:
        """Extract posts from a page's HTML.
        
        Args:
            soup: BeautifulSoup object of the page
            page_num: Current page number
            start_position: Starting global position for posts
            
        Returns:
            List of post dictionaries
        """
        posts = []
        
        # Try different post selectors from platform configuration
        post_elements = None
        post_selectors = self.platform_config.get('selectors', {}).get('posts', ['.post', '.message'])
        
        for selector in post_selectors:
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
                    post_elem, page_num, start_position + i, page_url
                )
                if post_data:
                    posts.append(post_data)
                    
            except Exception as e:
                logger.warning(f"Error extracting post {i} on page {page_num}: {e}")
                continue
        
        return posts
    
    def _extract_single_post(self, post_elem, page_num: int, global_position: int, base_url: str = None) -> Optional[Dict]:
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
            
            # Extract date and parse it
            date_raw = self._extract_date(post_elem)
            parsed_date = parse_forum_date(date_raw)
            
            # Extract post URL/ID
            post_url, post_id = self._extract_post_url(post_elem, base_url)
            
            # Extract vote counts
            vote_data = self._extract_votes(post_elem)
            
            # Create post data with enhanced metadata
            post_data = {
                'content': content_clean,
                'author': author,
                'date': date_raw,
                'parsed_date': parsed_date,
                'timestamp': parsed_date.timestamp() if parsed_date else 0,
                'page': page_num,
                'position_on_page': global_position - ((page_num - 1) * 20),  # Approximate
                'global_position': global_position,
                'url': post_url,
                'post_id': post_id,
                'hash': post_hash(content_clean, author, date_raw),
                **vote_data  # Include upvotes, downvotes, reactions
            }
            
            return post_data
            
        except Exception as e:
            logger.warning(f"Error extracting post data: {e}")
            return None
    
    def _extract_author(self, post_elem) -> str:
        """Extract author from post element using platform-specific selectors."""
        author_selectors = self.platform_config.get('selectors', {}).get('author', ['.author', '.username'])
        
        for selector in author_selectors:
            author_elem = post_elem.select_one(selector)
            if author_elem:
                author = author_elem.get_text(strip=True)
                if author and len(author) < 100:  # Sanity check
                    return author
        
        return 'unknown-author'
    
    def _extract_date(self, post_elem) -> str:
        """Extract date from post element using platform-specific selectors."""
        date_selectors = self.platform_config.get('selectors', {}).get('date', ['time', '.date', '.timestamp'])
        
        for selector in date_selectors:
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
    
    def _extract_post_url(self, post_elem, base_url: str = None) -> Tuple[str, str]:
        """Extract post URL and ID from post element.
        
        Args:
            post_elem: BeautifulSoup element containing the post
            
        Returns:
            Tuple of (post_url, post_id)
        """
        post_url = ''
        post_id = ''
        
        try:
            # Common patterns for post URLs and IDs
            url_selectors = [
                'a[href*="#post-"]',          # XenForo #post-123456
                'a[href*="#p"]',              # Various forums #p123456  
                'a[href*="post"]',            # Generic post links
                '.message-permalink',         # XenForo permalink
                '.post-permalink',            # Generic permalink
                '[data-post-id]',             # Data attribute
                '.postNumber a',              # Post number links
                '.post-link'                  # Generic post link class
            ]
            
            for selector in url_selectors:
                url_elem = post_elem.select_one(selector)
                if url_elem:
                    href = url_elem.get('href', '')
                    if href:
                        post_url = href
                        # Extract post ID from URL patterns like #post-123456 or #p123456
                        import re
                        id_match = re.search(r'#(?:post-|p)(\d+)', href)
                        if id_match:
                            post_id = id_match.group(1)
                        break
            
            # Try data attributes for post ID if not found in URL
            if not post_id:
                id_attrs = ['data-post-id', 'data-id', 'id']
                for attr in id_attrs:
                    attr_value = post_elem.get(attr, '')
                    if attr_value:
                        # Extract numbers from attribute value
                        import re
                        numbers = re.findall(r'\d+', attr_value)
                        if numbers:
                            post_id = numbers[0]
                            break
            
            # If we have a post ID but no URL, try to construct URL
            if post_id and not post_url and base_url:
                # Construct full URL with base thread URL
                if '#' in base_url:
                    base_url = base_url.split('#')[0]  # Remove any existing fragment
                post_url = f"{base_url}#post-{post_id}"
            elif post_id and not post_url:
                # Fallback if no base URL provided
                post_url = f"#post-{post_id}"
            
            # Convert relative URLs to absolute URLs
            if post_url and base_url and not post_url.startswith('http'):
                if post_url.startswith('#'):
                    # Fragment identifier - append to base URL
                    base_clean = base_url.split('#')[0]
                    post_url = f"{base_clean}{post_url}"
                else:
                    # Relative URL - use urljoin
                    post_url = urljoin(base_url, post_url)
                
        except Exception as e:
            logger.debug(f"Error extracting post URL: {e}")
        
        return post_url, post_id
    
    def _extract_votes(self, post_elem) -> Dict[str, int]:
        """Extract vote counts from post element.
        
        Args:
            post_elem: BeautifulSoup element containing the post
            
        Returns:
            Dictionary with vote counts
        """
        vote_data = {
            'upvotes': 0,
            'downvotes': 0,
            'reactions': 0,
            'likes': 0,
            'total_score': 0
        }
        
        try:
            # Try various vote selectors from platform configuration
            vote_selectors = self.platform_config.get('selectors', {}).get('votes', {})
            for vote_type, selectors in vote_selectors.items():
                for selector in selectors:
                    vote_elem = post_elem.select_one(selector)
                    if vote_elem:
                        # Extract numeric value
                        vote_text = vote_elem.get_text(strip=True)
                        # Handle various formats: "5", "+5", "▲5", "👍 5", etc.
                        import re
                        numbers = re.findall(r'\d+', vote_text)
                        if numbers:
                            vote_count = int(numbers[0])
                            vote_data[vote_type] = vote_count
                            break
            
            # Calculate total score (upvotes - downvotes + reactions/likes)
            vote_data['total_score'] = (
                vote_data['upvotes'] - vote_data['downvotes'] + 
                vote_data['reactions'] + vote_data['likes']
            )
            
            # For forums without explicit vote counts, look for reaction indicators
            if vote_data['total_score'] == 0:
                # Count reaction elements (emojis, thumbs up, etc.)
                reaction_indicators = [
                    '.reaction', '.like', '.thanks', '.agree', 
                    '👍', '❤️', '🔥', '.positive-reaction'
                ]
                reaction_count = 0
                for indicator in reaction_indicators:
                    if '👍' in indicator or '❤️' in indicator or '🔥' in indicator:
                        # Count emoji occurrences in text
                        reaction_count += post_elem.get_text().count(indicator)
                    else:
                        # Count elements
                        reaction_count += len(post_elem.select(indicator))
                
                vote_data['reactions'] = reaction_count
                vote_data['total_score'] = reaction_count
            
        except Exception as e:
            logger.debug(f"Error extracting votes: {e}")
        
        return vote_data
    
    def _find_next_page_url(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find the URL of the next page with enhanced detection.
        
        Args:
            soup: BeautifulSoup object of current page
            current_url: URL of current page
            
        Returns:
            URL of next page or None
        """
        # Enhanced patterns for next page links
        next_selectors = [
            # Standard pagination
            'a[rel="next"]',
            'link[rel="next"]',
            
            # Text-based selectors (more robust)
            'a[title*="Next"]',
            'a[aria-label*="Next"]',
            
            # Forum-specific patterns
            '.pagination a:contains("Next")',
            '.pagelinks a:contains("Next")',
            '.paginationContainer a:contains("Next")',
            '.pageNav a:contains("Next")',
            
            # Symbol-based navigation
            'a:contains(">")',
            'a:contains("»")',
            'a:contains("→")',
            
            # XenForo specific
            '.pageNav-jump--next',
            '.pageNav-jump[data-page-jump="next"]',
            
            # vBulletin specific
            '.pagination_next a',
            'a[title="Next Page"]',
            
            # phpBB specific
            '.pagination .next a',
        ]
        
        for selector in next_selectors:
            try:
                if ':contains(' in selector:
                    # Handle contains selectors specially
                    text_to_find = selector.split(':contains("')[1].split('")')[0]
                    links = soup.find_all('a')
                    for link in links:
                        if link.get_text(strip=True).lower() == text_to_find.lower():
                            if link.get('href'):
                                next_url = urljoin(current_url, link['href'])
                                if next_url != current_url:
                                    return next_url
                else:
                    next_elem = soup.select_one(selector)
                    if next_elem and next_elem.get('href'):
                        next_url = urljoin(current_url, next_elem['href'])
                        if next_url != current_url and self._is_valid_next_page(next_url, current_url):
                            return next_url
            except Exception as e:
                logger.debug(f"Error with selector {selector}: {e}")
                continue
        
        # Enhanced numbered pagination detection
        next_url = self._find_numbered_next_page(soup, current_url)
        if next_url:
            return next_url
        
        # JavaScript pagination fallback
        next_url = self._detect_js_pagination(soup, current_url)
        if next_url:
            return next_url
        
        return None
    
    def _is_valid_next_page(self, next_url: str, current_url: str) -> bool:
        """Validate that the next URL is actually a next page.
        
        Args:
            next_url: Candidate next page URL
            current_url: Current page URL
            
        Returns:
            True if valid next page
        """
        try:
            from urllib.parse import urlparse, parse_qs
            
            current_parsed = urlparse(current_url)
            next_parsed = urlparse(next_url)
            
            # Must be same domain
            if current_parsed.netloc != next_parsed.netloc:
                return False
            
            # Look for page number increases
            current_query = parse_qs(current_parsed.query)
            next_query = parse_qs(next_parsed.query)
            
            # Common page parameters
            page_params = ['page', 'p', 'start', 'offset']
            
            for param in page_params:
                if param in current_query and param in next_query:
                    try:
                        current_page = int(current_query[param][0])
                        next_page = int(next_query[param][0])
                        return next_page > current_page
                    except (ValueError, IndexError):
                        continue
            
            # Check for page numbers in path
            import re
            current_page_match = re.search(r'/page-?(\d+)', current_parsed.path)
            next_page_match = re.search(r'/page-?(\d+)', next_parsed.path)
            
            if current_page_match and next_page_match:
                return int(next_page_match.group(1)) > int(current_page_match.group(1))
            
            return True  # If we can't determine, assume it's valid
            
        except Exception:
            return True
    
    def _find_numbered_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Find next page using numbered pagination.
        
        Args:
            soup: BeautifulSoup object of current page
            current_url: Current page URL
            
        Returns:
            Next page URL or None
        """
        try:
            current_page_num = self._extract_current_page_number(soup)
            if current_page_num:
                # Look for next page number
                next_page_num = current_page_num + 1
                
                # Try various selectors for the next page number
                next_page_selectors = [
                    f'a:contains("{next_page_num}")',
                    f'a[href*="page={next_page_num}"]',
                    f'a[href*="p={next_page_num}"]',
                    f'a[href*="page-{next_page_num}"]'
                ]
                
                for selector in next_page_selectors:
                    if ':contains(' in selector:
                        # Handle contains selector
                        links = soup.find_all('a')
                        for link in links:
                            if link.get_text(strip=True) == str(next_page_num):
                                if link.get('href'):
                                    return urljoin(current_url, link['href'])
                    else:
                        next_link = soup.select_one(selector)
                        if next_link and next_link.get('href'):
                            return urljoin(current_url, next_link['href'])
        except Exception as e:
            logger.debug(f"Error in numbered pagination: {e}")
        
        return None
    
    def _detect_js_pagination(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        """Detect JavaScript-based pagination patterns.
        
        Args:
            soup: BeautifulSoup object of current page
            current_url: Current page URL
            
        Returns:
            Next page URL or None
        """
        try:
            # Look for data attributes that might contain next page info
            js_pagination_selectors = [
                '[data-page-next]',
                '[data-next-page]',
                '[data-href*="page"]',
                'button[onclick*="page"]',
                'a[data-page]'
            ]
            
            for selector in js_pagination_selectors:
                elem = soup.select_one(selector)
                if elem:
                    # Extract URL from data attributes
                    for attr in ['data-page-next', 'data-next-page', 'data-href']:
                        url = elem.get(attr)
                        if url:
                            return urljoin(current_url, url)
                    
                    # Extract from onclick handlers
                    onclick = elem.get('onclick', '')
                    if 'page' in onclick:
                        import re
                        url_match = re.search(r'["\']([^"\']*page[^"\']*)["\']', onclick)
                        if url_match:
                            return urljoin(current_url, url_match.group(1))
        
        except Exception as e:
            logger.debug(f"Error detecting JS pagination: {e}")
        
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
        """Get comprehensive scraping statistics."""
        duration = time.time() - self.stats['start_time']
        active_scraping_time = duration - self.stats['total_delay_time']
        
        return {
            **self.stats,
            'duration': duration,
            'active_scraping_time': active_scraping_time,
            'delay_percentage': (self.stats['total_delay_time'] / max(1, duration)) * 100,
            'success_rate': (
                self.stats['successful_requests'] / max(1, self.stats['total_requests'])
            ),
            'posts_per_second': self.stats['total_posts_scraped'] / max(1, active_scraping_time),
            'requests_per_minute': (self.stats['total_requests'] / max(1, duration)) * 60,
            'average_delay_per_request': self.stats['total_delay_time'] / max(1, self.stats['total_requests'])
        }


__all__ = ['ForumScraper']
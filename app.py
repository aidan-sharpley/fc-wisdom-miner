import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import hnswlib
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, Response, jsonify, render_template, request
from tqdm import tqdm

# -------------------- Configuration --------------------
BASE_TMP_DIR = os.environ.get("BASE_TMP_DIR", "tmp")
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_API_URL = os.environ.get(
    "OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
INDEX_META_NAME = "index_meta.pkl"
HNSW_INDEX_NAME = "index_hnsw.bin"
METADATA_INDEX_NAME = "metadata_index.json"
CACHE_PATH = os.path.join(BASE_TMP_DIR, "embeddings_cache.pkl")

# Performance tuning
API_TIMEOUT = 15  # Increased timeout
MAX_RETRIES = 3
RETRY_BACKOFF = 2
CHUNK_SIZE = 800  # Slightly smaller chunks for better embedding quality
CHUNK_OVERLAP = 150
QUERY_RERANK_SIZE = 20  # Reduced for faster processing
BATCH_RERANK_TIMEOUT = 45  # Increased for complex queries
FINAL_TOP_K = 7
MAX_WORKERS = 4  # Limit concurrent embedding requests

# HNSW parameters for better search quality
HNSW_M = 16  # Lower M for faster search
HNSW_EF_CONSTRUCTION = 100  # Lower for faster index building
HNSW_EF_SEARCH = 50  # Search parameter

# Expected embedding dimension (adjust based on your model)
EXPECTED_EMBED_DIM = 768  # Common dimension for many embedding models

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASE_TMP_DIR, "app.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-secure-secret")


# -------------------- Utilities --------------------
def post_hash(content: str) -> str:
    """Create a consistent hash for caching embeddings."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    """Ensure URL has proper schema."""
    return url if url.startswith(("http://", "https://")) else "https://" + url


def get_thread_dir(thread_key: str) -> str:
    """Create and return thread directory path."""
    # Sanitize thread key
    thread_key = re.sub(r"[^\w\-_.]", "_", thread_key)
    path = os.path.join(BASE_TMP_DIR, thread_key)
    os.makedirs(path, exist_ok=True)
    return path


def list_threads() -> List[str]:
    """List all available thread directories."""
    if not os.path.isdir(BASE_TMP_DIR):
        return []
    return sorted(
        d
        for d in os.listdir(BASE_TMP_DIR)
        if os.path.isdir(os.path.join(BASE_TMP_DIR, d)) and d != "__pycache__"
    )


def clean_post_content(raw: str) -> str:
    """Clean HTML and forum-specific formatting from post content."""
    if not raw:
        return ""

    soup = BeautifulSoup(raw, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    text = soup.get_text()

    # Remove forum-specific patterns
    text = re.sub(r"(?:^|\n)\s*\w+ said:\s*(Click to expand\.{3})?", "", text)
    text = re.sub(r"Click to expand\.{3}", "", text)
    text = re.sub(r"Quote:\s*", "", text)
    text = re.sub(r"Originally posted by.*?:", "", text)

    # Normalize whitespace
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


# -------------------- HTML Parsing --------------------
def extract_date(el) -> str:
    """Extract post date from HTML element."""
    # Try multiple selectors for different forum formats
    selectors = ["time[datetime]", ".message-date time", ".postDate", "[data-time]"]

    for selector in selectors:
        tag = el.select_one(selector)
        if tag:
            if tag.has_attr("datetime"):
                return tag["datetime"]
            elif tag.has_attr("data-time"):
                return tag["data-time"]
            elif tag.text:
                return tag.text.strip()

    return "unknown-date"


def extract_author(el) -> str:
    """Extract post author from HTML element."""
    # Try data attribute first
    if el.has_attr("data-author"):
        return el["data-author"]

    # Try various selectors
    selectors = [
        ".message-name .username",
        ".author .username",
        ".postbit_legacy .bigusername",
        ".postauthor .username",
    ]

    for selector in selectors:
        author_el = el.select_one(selector)
        if author_el:
            return author_el.get_text().strip()

    return "unknown-author"


def extract_content(el) -> str:
    """Extract post content from HTML element."""
    # Try multiple content selectors
    selectors = [
        "div.message-userContent .bbWrapper",
        ".message-body .bbWrapper",
        ".postbody .content",
        ".post_message",
        ".message-content",
    ]

    for selector in selectors:
        node = el.select_one(selector)
        if node:
            return node.get_text(separator="\n").strip()

    # Fallback to element text
    return el.get_text(separator="\n").strip()


def extract_post_url(el, base: str) -> str:
    """Extract individual post URL."""
    # Try to find permalink
    link = el.select_one(
        "ul.message-attribution-opposite a[href], .postbit_legacy .postdate a"
    )
    if link and link.has_attr("href"):
        return requests.compat.urljoin(base, link["href"])

    # Try data-content attribute
    data = el.get("data-content", "")
    if data.startswith("post-"):
        pid = data.split("-")[1]
        return f"{base}posts/{pid}/"

    # Try to extract post ID from various attributes
    post_id = (
        el.get("id", "").replace("post_", "").replace("post", "")
        or el.get("data-post-id", "")
        or el.get("data-pid", "")
    )

    if post_id.isdigit():
        return f"{base}posts/{post_id}/"

    return f"{base}#unknown-post"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    """Extract canonical URL from page."""
    link = soup.find("link", rel="canonical")
    if link and link.has_attr("href"):
        url = link["href"]
        # Remove page number from URL
        return re.sub(r"page-\d+/?$", "", url).rstrip("/") + "/"
    return ""


# -------------------- Web Scraping --------------------
def detect_last_page(url: str) -> int:
    """Detect the last page number of a forum thread."""
    try:
        logger.info(f"Detecting last page for: {url}")
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        # Try pagination input
        inp = soup.select_one("input.js-pageJumpPage[max]")
        if inp and inp.has_attr("max"):
            last_page = int(inp["max"])
            logger.info(f"Found last page from input: {last_page}")
            return last_page

        # Try pagination links
        last = soup.select_one(".pageNav-main li:last-child a")
        if last and last.text.strip().isdigit():
            last_page = int(last.text.strip())
            logger.info(f"Found last page from navigation: {last_page}")
            return last_page

        # Try other pagination patterns
        page_links = soup.select(".pageNumbers a, .pagination a")
        if page_links:
            page_nums = [
                int(link.text) for link in page_links if link.text.strip().isdigit()
            ]
            if page_nums:
                last_page = max(page_nums)
                logger.info(f"Found last page from links: {last_page}")
                return last_page

        logger.warning(f"Could not detect pagination for {url}, defaulting to 1")
    except Exception as e:
        logger.error(f"Error detecting last page for {url}: {e}")

    return 1


def fetch_forum_pages(base_url: str, last_page: int, save_dir: str):
    """Fetch all pages of a forum thread."""
    base = re.sub(r"page-\d+/?$", "", base_url).rstrip("/")

    logger.info(f"Fetching {last_page} pages from {base}")

    for i in range(1, last_page + 1):
        fn = os.path.join(save_dir, f"page{i}.html")
        if os.path.exists(fn):
            logger.debug(f"Page {i} already exists, skipping")
            continue

        try:
            # Construct URL for page
            page_url = f"{base}/page-{i}" if i > 1 else base

            r = requests.get(page_url, timeout=API_TIMEOUT)
            r.raise_for_status()

            with open(fn, "w", encoding="utf-8") as f:
                f.write(r.text)

            logger.info(f"Saved page {i}/{last_page}")
            time.sleep(0.5)  # Be respectful to the server

        except Exception as e:
            logger.error(f"Error fetching page {i}: {e}")
            break


# -------------------- HTTP with Retries --------------------
def http_post(url: str, json_data: dict, timeout: int) -> requests.Response:
    """Make HTTP POST request with retries."""
    last_exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=json_data, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_exception = e
            logger.warning(
                f"Request to {url} failed (attempt {attempt}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES:
                sleep_time = RETRY_BACKOFF ** (attempt - 1)
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

    raise last_exception


# -------------------- Embedding --------------------
def load_cache() -> Dict[str, np.ndarray]:
    """Load embedding cache from disk."""
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                cache = pickle.load(f)
            return cache
        except Exception as e:
            logger.warning(f"Embeddings cache corrupted ({e}); resetting.")
    return {}


def save_cache(cache: Dict[str, np.ndarray]):
    """Save embedding cache to disk."""
    try:
        os.makedirs(BASE_TMP_DIR, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
        logger.debug(f"Saved {len(cache)} embeddings to cache")
    except Exception as e:
        logger.error(f"Failed to save embedding cache: {e}")


def chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    if not text:
        return [""]

    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end]

        # Try to break at sentence boundaries
        if end < length and "." in chunk[-50:]:
            last_period = chunk.rfind(".", max(0, len(chunk) - 50))
            if last_period > len(chunk) // 2:
                end = start + last_period + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())

        if end >= length:
            break

        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0

    return chunks


def embed_text(text: str) -> np.ndarray:
    """Generate embedding for text with caching and error handling."""
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding")
        return np.zeros(EXPECTED_EMBED_DIM, dtype=np.float32)

    cache = load_cache()
    h = post_hash(text.strip())

    if h in cache:
        embedding = cache[h]
        if embedding.size > 0:
            return embedding
        else:
            logger.warning(f"Found empty embedding in cache for hash {h[:10]}...")

    chunks = chunk_text(text.strip())
    if not chunks:
        chunks = [text.strip()]

    vectors = []

    for chunk in chunks:
        if not chunk.strip():
            continue

        try:
            r = http_post(
                OLLAMA_EMBED_API_URL,
                {"model": OLLAMA_EMBED_MODEL, "prompt": chunk.strip()},
                API_TIMEOUT,
            )

            response_data = r.json()
            embedding_data = response_data.get("embedding", [])

            if not embedding_data:
                logger.error(f"No embedding data in response: {response_data}")
                continue

            emb = np.array(embedding_data, dtype=np.float32)

            if emb.size == 0:
                logger.error(f"Empty embedding array for chunk: {chunk[:50]}...")
                continue

            vectors.append(emb)

        except Exception as e:
            logger.error(f"Failed to embed chunk '{chunk[:50]}...': {e}")
            continue

    if not vectors:
        logger.error(f"No valid embeddings generated for text: {text[:100]}...")
        return np.zeros(EXPECTED_EMBED_DIM, dtype=np.float32)

    # Average the chunk embeddings
    final_embedding = np.mean(vectors, axis=0)

    # Validate embedding
    if final_embedding.size == 0:
        logger.error("Final embedding is empty")
        return np.zeros(EXPECTED_EMBED_DIM, dtype=np.float32)

    # Cache the result
    cache[h] = final_embedding
    save_cache(cache)

    logger.debug(
        f"Generated embedding of dimension {final_embedding.shape[0]} for text hash {h[:10]}..."
    )
    return final_embedding


# -------------------- Preprocessing --------------------
def preprocess_thread(thread_dir: str, force: bool = False):
    """Preprocess forum thread data for search."""
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    idx_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    meta_index = os.path.join(thread_dir, METADATA_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")

    # Check if preprocessing already exists
    if not force and all(
        os.path.exists(p) for p in [meta_path, idx_path, meta_index, posts_dir]
    ):
        logger.info(f"Preprocessing exists for {thread_dir}, skipping.")
        return

    logger.info(f"Preprocessing thread in {thread_dir} (force={force})")

    # Clean up if forcing rebuild
    if force and os.path.exists(posts_dir):
        shutil.rmtree(posts_dir)
    os.makedirs(posts_dir, exist_ok=True)

    # Process HTML files
    html_files = sorted(f for f in os.listdir(thread_dir) if f.endswith(".html"))
    if not html_files:
        logger.error(f"No HTML files found in {thread_dir}")
        return

    raw_posts = []
    metadata = []

    logger.info(f"Processing {len(html_files)} HTML files")

    for fn in html_files:
        try:
            page_match = re.search(r"page(\d+)", fn)
            page = int(page_match.group(1)) if page_match else 1

            with open(os.path.join(thread_dir, fn), encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")

            base = extract_canonical_url(soup) or "https://forum.example.com/"

            # Find post elements - try multiple selectors
            post_selectors = [
                "article.message",
                ".post",
                ".postbit_legacy",
                "[data-post-id]",
            ]

            posts_found = False
            for selector in post_selectors:
                elements = soup.select(selector)
                if elements:
                    posts_found = True
                    logger.debug(
                        f"Found {len(elements)} posts using selector '{selector}' on page {page}"
                    )

                    for el in elements:
                        content = clean_post_content(extract_content(el))
                        if (
                            not content or len(content.strip()) < 10
                        ):  # Skip very short posts
                            continue

                        post = {
                            "page": page,
                            "date": extract_date(el),
                            "author": extract_author(el),
                            "content": content,
                            "url": extract_post_url(el, base),
                        }
                        raw_posts.append(post)
                        metadata.append(
                            {
                                "page": page,
                                "date": post["date"],
                                "author": post["author"],
                            }
                        )
                    break

            if not posts_found:
                logger.warning(f"No posts found on page {page} using any selector")

        except Exception as e:
            logger.error(f"Error processing {fn}: {e}")

    if not raw_posts:
        logger.error("No posts extracted from HTML files")
        return

    logger.info(f"Extracted {len(raw_posts)} posts")

    # Save metadata
    with open(meta_index, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Generate embeddings
    cache = load_cache()
    to_embed = []

    for post in raw_posts:
        h = post_hash(post["content"])
        if force or h not in cache or cache.get(h, np.array([])).size == 0:
            to_embed.append(post)

    logger.info(
        f"Embedding {len(to_embed)} posts (cached: {len(raw_posts) - len(to_embed)})"
    )

    if to_embed:
        # Embed posts with limited concurrency
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_post = {
                executor.submit(embed_text, post["content"]): post for post in to_embed
            }

            for future in tqdm(
                as_completed(future_to_post),
                total=len(future_to_post),
                desc="Embedding posts",
            ):
                post = future_to_post[future]
                try:
                    embedding = future.result()
                    if embedding.size == 0:
                        logger.error(f"Empty embedding for post by {post['author']}")
                    cache[post_hash(post["content"])] = embedding
                except Exception as e:
                    logger.error(f"Embedding failed for post by {post['author']}: {e}")

        save_cache(cache)

    # Build search index
    vectors = []
    valid_posts = []

    for i, post in enumerate(raw_posts):
        h = post_hash(post["content"])
        if h in cache:
            vec = cache[h]
            if vec.size > 0:
                vectors.append(vec)
                valid_posts.append(post)

                # Save individual post
                with open(
                    os.path.join(posts_dir, f"{len(valid_posts) - 1}.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(post, f, indent=2)
            else:
                logger.warning(f"Skipping post {i} due to empty embedding")
        else:
            logger.warning(f"No embedding found for post {i}")

    if not vectors:
        logger.error("No valid embeddings to build index")
        return

    # Validate embedding dimensions
    embedding_dims = [v.shape[0] for v in vectors]
    if len(set(embedding_dims)) > 1:
        logger.error(f"Inconsistent embedding dimensions: {set(embedding_dims)}")
        return

    dim = vectors[0].shape[0]
    logger.info(f"Building HNSW index with {len(vectors)} vectors of dimension {dim}")

    # Build HNSW index
    try:
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(
            max_elements=len(vectors), ef_construction=HNSW_EF_CONSTRUCTION, M=HNSW_M
        )
        index.add_items(np.vstack(vectors))
        index.set_ef(HNSW_EF_SEARCH)  # Set search parameter
        index.save_index(idx_path)

        # Save metadata
        with open(meta_path, "wb") as f:
            pickle.dump({"dim": dim, "count": len(vectors)}, f)

        logger.info(f"Built HNSW index with {len(vectors)} items (dimension {dim})")

    except Exception as e:
        logger.error(f"Failed to build HNSW index: {e}")
        raise


# -------------------- HyDE --------------------
def generate_hyde(query: str) -> str:
    """Generate hypothetical document for better retrieval."""
    prompt = (
        "Write a detailed forum post that would contain the answer to this question. "
        "Include technical details, user experiences, and specific recommendations.\n\n"
        f"Question: {query}\n\n"
        "Forum post:"
    )

    try:
        r = http_post(
            OLLAMA_API_URL,
            {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "options": {"temperature": 0.3, "max_tokens": 200},
            },
            API_TIMEOUT,
        )

        response_text = r.text.strip()
        if not response_text:
            return query

        # Try to parse JSON response
        try:
            data = json.loads(response_text)
            generated = data.get("response", "").strip()
            if generated:
                logger.debug(f"Generated HyDE: {generated[:100]}...")
                return generated
        except json.JSONDecodeError:
            pass

        # Try parsing streaming response
        lines = response_text.strip().split("\n")
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                response = obj.get("response", "").strip()
                if response:
                    logger.debug(f"Generated HyDE from stream: {response[:100]}...")
                    return response
            except json.JSONDecodeError:
                continue

        # Fallback: use raw response
        if response_text:
            logger.debug(f"Using raw HyDE response: {response_text[:100]}...")
            return response_text

    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")

    return query


# -------------------- Rerank --------------------
def batch_rerank(query: str, posts: List[Dict]) -> List[Tuple[int, Dict]]:
    """Rerank posts using LLM scoring."""
    if not posts:
        return []

    # Create scoring prompt
    lines = [
        f"Question: {query}",
        "",
        "Rate how well each post answers the question (0-10, where 10 is perfect match):",
        "Format: Post X: [score]",
        "",
    ]

    for i, post in enumerate(posts, 1):
        preview = post["content"][:400].replace("\n", " ").strip()
        if len(post["content"]) > 400:
            preview += "..."
        lines.append(f"Post {i}: {preview}")
        lines.append("")

    prompt = "\n".join(lines) + "\nScores:"

    try:
        r = http_post(
            OLLAMA_API_URL,
            {"model": OLLAMA_MODEL, "prompt": prompt, "options": {"temperature": 0.1}},
            BATCH_RERANK_TIMEOUT,
        )

        response_text = r.text.strip()

        # Try to parse JSON
        try:
            data = json.loads(response_text)
            text = data.get("response", "")
        except json.JSONDecodeError:
            # Try parsing streaming response
            text = ""
            for line in response_text.split("\n"):
                if line.strip():
                    try:
                        obj = json.loads(line)
                        text += obj.get("response", "")
                    except json.JSONDecodeError:
                        text = response_text
                        break

        # Extract scores
        scores = []
        for match in re.finditer(
            r"Post\s+(\d+):\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE
        ):
            post_num = int(match.group(1))
            score = float(match.group(2))
            if 1 <= post_num <= len(posts):
                scores.append((post_num - 1, score))  # Convert to 0-based index

        # If we don't have scores for all posts, fill with defaults
        scored_indices = {idx for idx, _ in scores}
        for i in range(len(posts)):
            if i not in scored_indices:
                scores.append((i, 1.0))  # Low default score

        # Sort by score and return
        scores.sort(key=lambda x: x[1], reverse=True)
        result = [(int(score), posts[idx]) for idx, score in scores]

        logger.debug(f"Reranking scores: {[score for score, _ in result[:5]]}")
        return result

    except Exception as e:
        logger.warning(f"Rerank failed: {e}")
        # Return posts with default scores
        return [(5, post) for post in posts]


# -------------------- Search --------------------
def find_posts_by_metadata(
    thread_dir: str,
    author: Optional[str] = None,
    page: Optional[int] = None,
    date: Optional[str] = None,
) -> List[Dict]:
    """Find posts by metadata criteria."""
    meta_path = os.path.join(thread_dir, METADATA_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")

    if not os.path.exists(meta_path):
        logger.warning(f"Metadata index not found: {meta_path}")
        return []

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return []

    matching_indices = []

    for i, meta in enumerate(metadata):
        match = True

        if author and author.lower() not in meta.get("author", "").lower():
            match = False
        if page and page != meta.get("page"):
            match = False
        if date and date.lower() not in meta.get("date", "").lower():
            match = False

        if match:
            matching_indices.append(i)

    # Load matching posts
    posts = []
    for i in matching_indices[:15]:  # Limit to 15 results
        post_path = os.path.join(posts_dir, f"{i}.json")
        try:
            with open(post_path, "r", encoding="utf-8") as f:
                posts.append(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load post {i}: {e}")

    logger.info(f"Found {len(posts)} posts matching metadata criteria")
    return posts


class QueryType(Enum):
    SPECIFIC_POST = "specific_post"  # "second post", "first post"
    AUTHOR_SEARCH = "author_search"  # "posts by user"
    TEMPORAL_SEARCH = "temporal_search"  # "posts from yesterday"
    CONTENT_SEARCH = "content_search"  # semantic content search
    METADATA_SEARCH = "metadata_search"  # page, date, etc.
    COMPARATIVE = "comparative"  # "compare X and Y"
    SUMMARY = "summary"  # "summarize the thread"


@dataclass
class QueryIntent:
    """Structured representation of user query intent"""

    query_type: QueryType
    target: Optional[str] = None  # "second", "first", username, etc.
    filters: Dict[str, Any] = None  # page, date, author filters
    semantic_query: Optional[str] = None  # refined query for semantic search
    confidence: float = 0.0

    def __post_init__(self):
        if self.filters is None:
            self.filters = {}


class IntelligentQueryProcessor:
    """Advanced query processor that understands natural language intent"""

    def __init__(self):
        # Ordinal patterns
        self.ordinal_patterns = {
            r"\b(first|1st)\b": 1,
            r"\b(second|2nd)\b": 2,
            r"\b(third|3rd)\b": 3,
            r"\b(fourth|4th)\b": 4,
            r"\b(fifth|5th)\b": 5,
            r"\b(sixth|6th)\b": 6,
            r"\b(seventh|7th)\b": 7,
            r"\b(eighth|8th)\b": 8,
            r"\b(ninth|9th)\b": 9,
            r"\b(tenth|10th)\b": 10,
            r"\b(last|final)\b": -1,
            r"\b(latest|most recent)\b": -1,
        }

        # Time patterns
        self.time_patterns = {
            r"\b(today|this morning|this afternoon|this evening)\b": "today",
            r"\b(yesterday|last night)\b": "yesterday",
            r"\b(this week|past week)\b": "week",
            r"\b(this month|past month)\b": "month",
            r"\b(january|jan)\b": "january",
            r"\b(february|feb)\b": "february",
            r"\b(march|mar)\b": "march",
            r"\b(april|apr)\b": "april",
            r"\b(may)\b": "may",
            r"\b(june|jun)\b": "june",
            r"\b(july|jul)\b": "july",
            r"\b(august|aug)\b": "august",
            r"\b(september|sep|sept)\b": "september",
            r"\b(october|oct)\b": "october",
            r"\b(november|nov)\b": "november",
            r"\b(december|dec)\b": "december",
            r"\b(20\d{2})\b": "year",
        }

        # Author patterns
        self.author_patterns = [
            r"\bposts?\s+by\s+([^\s,\.]+)",
            r"\b([^\s,\.]+)\s+(?:said|wrote|posted)",
            r"\bauthor:?\s*([^\s,\.]+)",
            r"\buser:?\s*([^\s,\.]+)",
            r"\bfrom\s+([^\s,\.]+)",
        ]

    def analyze_query(self, query: str) -> QueryIntent:
        """Analyze query and determine intent with structured extraction"""
        query_lower = query.lower().strip()

        # Try specific post patterns first
        post_intent = self._detect_specific_post(query_lower)
        if post_intent.confidence > 0.7:
            return post_intent

        # Try author search
        author_intent = self._detect_author_search(query_lower)
        if author_intent.confidence > 0.6:
            return author_intent

        # Try temporal search
        temporal_intent = self._detect_temporal_search(query_lower)
        if temporal_intent.confidence > 0.6:
            return temporal_intent

        # Try metadata search
        metadata_intent = self._detect_metadata_search(query_lower)
        if metadata_intent.confidence > 0.5:
            return metadata_intent

        # Check for summary request
        if any(
            word in query_lower for word in ["summary", "summarize", "overview", "tldr"]
        ):
            return QueryIntent(
                query_type=QueryType.SUMMARY, semantic_query=query, confidence=0.8
            )

        # Default to semantic content search
        return QueryIntent(
            query_type=QueryType.CONTENT_SEARCH, semantic_query=query, confidence=0.3
        )

    def _detect_specific_post(self, query: str) -> QueryIntent:
        """Detect requests for specific posts (first, second, last, etc.)"""
        confidence = 0.0
        target_position = None

        # Look for ordinal indicators
        for pattern, position in self.ordinal_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                target_position = position
                confidence = 0.9
                break

        # Look for "Nth post" patterns
        if not target_position:
            match = re.search(r"\b(\d+)(?:st|nd|rd|th)?\s+post", query)
            if match:
                target_position = int(match.group(1))
                confidence = 0.8

        # Check for post-related keywords to boost confidence
        post_keywords = ["post", "message", "reply", "comment"]
        if any(keyword in query for keyword in post_keywords):
            confidence += 0.1

        # Extract additional context
        filters = {}
        page_match = re.search(r"(?:on|from|in)\s+page\s+(\d+)", query)
        if page_match:
            filters["page"] = int(page_match.group(1))
            confidence += 0.1

        return QueryIntent(
            query_type=QueryType.SPECIFIC_POST,
            target=str(target_position) if target_position else None,
            filters=filters,
            confidence=confidence,
        )

    def _detect_author_search(self, query: str) -> QueryIntent:
        """Detect requests for posts by specific authors"""
        confidence = 0.0
        author = None

        for pattern in self.author_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                confidence = 0.8
                break

        # Look for @ mentions
        if not author:
            match = re.search(r"@([^\s,\.]+)", query)
            if match:
                author = match.group(1)
                confidence = 0.7

        return QueryIntent(
            query_type=QueryType.AUTHOR_SEARCH, target=author, confidence=confidence
        )

    def _detect_temporal_search(self, query: str) -> QueryIntent:
        """Detect time-based search requests"""
        confidence = 0.0
        time_filter = None

        for pattern, time_ref in self.time_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                time_filter = time_ref
                confidence = 0.7
                break

        # Look for specific dates
        date_match = re.search(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", query)
        if date_match:
            time_filter = date_match.group(1)
            confidence = 0.9

        return QueryIntent(
            query_type=QueryType.TEMPORAL_SEARCH,
            filters={"date": time_filter} if time_filter else {},
            semantic_query=query,
            confidence=confidence,
        )

    def _detect_metadata_search(self, query: str) -> QueryIntent:
        """Detect metadata-based searches (page, thread info, etc.)"""
        confidence = 0.0
        filters = {}

        # Page search
        page_match = re.search(r"(?:on|from|in)\s+page\s+(\d+)", query)
        if page_match:
            filters["page"] = int(page_match.group(1))
            confidence = 0.6

        # Thread structure questions
        if any(
            phrase in query
            for phrase in ["how many posts", "post count", "thread length"]
        ):
            confidence = 0.8

        return QueryIntent(
            query_type=QueryType.METADATA_SEARCH,
            filters=filters,
            semantic_query=query,
            confidence=confidence,
        )


def create_enhanced_system_prompt(intent: QueryIntent, context: str) -> str:
    """Create a more targeted system prompt based on query intent"""

    base_instructions = (
        "You are an expert forum analyst. Analyze the provided forum posts and answer the user's question accurately. "
        "Use ONLY the information from the provided posts. Be specific and cite relevant details.\n\n"
    )

    if intent.query_type == QueryType.SPECIFIC_POST:
        if intent.target:
            position = intent.target
            if position == "-1":
                specific_instruction = (
                    "The user is asking about the LAST post in the thread. "
                    "Find the chronologically last post and provide its details (date, author, content summary).\n"
                )
            else:
                specific_instruction = (
                    f"The user is asking about the {position} post in chronological order. "
                    f"Count the posts from the beginning and identify post #{position}. "
                    "Provide the date, author, and relevant details for that specific post.\n"
                )
        else:
            specific_instruction = (
                "The user is asking about a specific post position. "
                "Analyze the posts carefully to identify which one they're referring to.\n"
            )

    elif intent.query_type == QueryType.AUTHOR_SEARCH:
        specific_instruction = (
            f"Focus on posts by author '{intent.target}'. "
            "List their posts chronologically with dates and key points.\n"
        )

    elif intent.query_type == QueryType.TEMPORAL_SEARCH:
        specific_instruction = (
            "Focus on the temporal aspect of the query. "
            "Pay attention to post dates and chronological order.\n"
        )

    elif intent.query_type == QueryType.SUMMARY:
        specific_instruction = (
            "Provide a comprehensive summary of the thread. "
            "Include key points, main participants, and the evolution of the discussion.\n"
        )

    else:
        specific_instruction = (
            "Use semantic understanding to find the most relevant information. "
            "Consider the context and intent behind the user's question.\n"
        )

    return f"{base_instructions}{specific_instruction}---FORUM CONTEXT---\n{context}\n---END CONTEXT---"


def enhanced_post_retrieval(
    intent: QueryIntent, thread_dir: str, metadata: List[Dict]
) -> List[Dict]:
    """Enhanced post retrieval based on query intent"""

    if intent.query_type == QueryType.SPECIFIC_POST and intent.target:
        return retrieve_specific_post(intent, thread_dir, metadata)

    elif intent.query_type == QueryType.AUTHOR_SEARCH and intent.target:
        return retrieve_posts_by_author(intent.target, thread_dir, metadata)

    elif intent.query_type == QueryType.TEMPORAL_SEARCH:
        return retrieve_posts_by_time(intent, thread_dir, metadata)

    elif intent.query_type == QueryType.METADATA_SEARCH:
        return retrieve_posts_by_metadata(intent, thread_dir, metadata)

    # For content search, we'll still use the existing semantic search
    return []


def retrieve_specific_post(
    intent: QueryIntent, thread_dir: str, metadata: List[Dict]
) -> List[Dict]:
    """Retrieve a specific post by position"""
    try:
        position = int(intent.target)

        # Handle negative indexing (last post)
        if position == -1:
            position = len(metadata)

        # Convert to 0-based index
        index = position - 1

        if 0 <= index < len(metadata):
            post_path = os.path.join(thread_dir, "posts", f"{index}.json")
            if os.path.exists(post_path):
                with open(post_path, "r", encoding="utf-8") as f:
                    post = json.load(f)
                return [post]

    except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error retrieving specific post: {e}")

    return []


def retrieve_posts_by_author(
    author: str, thread_dir: str, metadata: List[Dict]
) -> List[Dict]:
    """Retrieve posts by specific author"""
    posts = []

    for i, meta in enumerate(metadata):
        if author.lower() in meta.get("author", "").lower():
            post_path = os.path.join(thread_dir, "posts", f"{i}.json")
            try:
                with open(post_path, "r", encoding="utf-8") as f:
                    post = json.load(f)
                posts.append(post)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

    return posts[:10]  # Limit results


def retrieve_posts_by_time(
    intent: QueryIntent, thread_dir: str, metadata: List[Dict]
) -> List[Dict]:
    """Retrieve posts by time criteria"""
    time_filter = intent.filters.get("date", "")
    posts = []

    for i, meta in enumerate(metadata):
        post_date = meta.get("date", "").lower()
        if time_filter.lower() in post_date:
            post_path = os.path.join(thread_dir, "posts", f"{i}.json")
            try:
                with open(post_path, "r", encoding="utf-8") as f:
                    post = json.load(f)
                posts.append(post)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

    return posts[:10]


def retrieve_posts_by_metadata(
    intent: QueryIntent, thread_dir: str, metadata: List[Dict]
) -> List[Dict]:
    """Retrieve posts by metadata filters"""
    posts = []

    for i, meta in enumerate(metadata):
        match = True

        for filter_key, filter_value in intent.filters.items():
            if filter_key == "page" and meta.get("page") != filter_value:
                match = False
                break

        if match:
            post_path = os.path.join(thread_dir, "posts", f"{i}.json")
            try:
                with open(post_path, "r", encoding="utf-8") as f:
                    post = json.load(f)
                posts.append(post)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

    return posts[:15]


# Example usage integration
def process_intelligent_query(
    query: str, thread_dir: str
) -> Tuple[QueryIntent, List[Dict]]:
    """Main function to process a query intelligently"""

    # Initialize processor
    processor = IntelligentQueryProcessor()

    # Analyze the query
    intent = processor.analyze_query(query)
    logger.info(
        f"Query intent: {intent.query_type.value}, confidence: {intent.confidence}"
    )

    # Load metadata
    metadata_path = os.path.join(thread_dir, "metadata_index.json")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = []

    # Retrieve posts based on intent
    posts = enhanced_post_retrieval(intent, thread_dir, metadata)

    return intent, posts


def find_relevant_posts(
    query: str, thread_dir: str, top_k: int = FINAL_TOP_K
) -> List[Dict]:
    """Find relevant posts using vector similarity search."""
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    idx_path = os.path.join(thread_dir, HNSW_INDEX_NAME)

    if not os.path.exists(meta_path) or not os.path.exists(idx_path):
        logger.error(f"Index files not found in {thread_dir}")
        return []

    try:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        count = meta.get("count", 0)
        dim = meta.get("dim", 0)

        if count == 0 or dim == 0:
            logger.error(f"Invalid index metadata: count={count}, dim={dim}")
            return []

        logger.debug(f"Loading index with {count} items, dimension {dim}")

        # Load HNSW index
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(idx_path)
        index.set_ef(HNSW_EF_SEARCH)

        # Generate query embedding with HyDE
        hyde_query = generate_hyde(query)
        query_embedding = embed_text(hyde_query)

        if query_embedding.size == 0:
            logger.error("Failed to generate query embedding")
            return []

        if query_embedding.size != dim:
            logger.error(
                f"Query embedding dim {query_embedding.size} != index dim {dim}"
            )
            return []

        # Search for similar posts
        k = min(QUERY_RERANK_SIZE, count)
        labels, distances = index.knn_query(query_embedding.reshape(1, -1), k=k)

        logger.debug(
            f"Found {len(labels[0])} candidate posts, distances: {distances[0][:5]}"
        )

        # Load candidate posts
        posts_dir = os.path.join(thread_dir, "posts")
        candidates = []

        for i, label in enumerate(labels[0]):
            post_path = os.path.join(posts_dir, f"{label}.json")
            if os.path.exists(post_path):
                try:
                    with open(post_path, "r", encoding="utf-8") as f:
                        post = json.load(f)
                    post["_similarity_score"] = float(
                        1.0 - distances[0][i]
                    )  # Convert distance to similarity
                    candidates.append(post)
                except Exception as e:
                    logger.warning(f"Failed to load post {label}: {e}")

        if not candidates:
            logger.warning("No candidate posts loaded successfully")
            return []

        logger.info(f"Loaded {len(candidates)} candidate posts for reranking")

        # Rerank using LLM
        scored_posts = batch_rerank(query, candidates)

        # Return top results
        final_posts = [post for _, post in scored_posts[:top_k]]
        logger.info(f"Returning {len(final_posts)} top-ranked posts")

        return final_posts

    except Exception as e:
        logger.error(f"Error in find_relevant_posts: {e}")
        return []


# -------------------- Routes --------------------
@app.route("/")
def index_route():
    """Main page showing available threads."""
    threads = list_threads()
    return render_template("index.html", threads=threads)


@app.route("/preprocess", methods=["POST"])
def preprocess_route():
    """Preprocess a forum thread for search."""
    try:
        data = request.json or {}
        thread_key = data.get("thread_key", "").strip()
        url = data.get("url", "").strip()
        refresh = data.get("refresh", False)

        if not thread_key and not url:
            return jsonify({"error": "thread_key or url required"}), 400

        # Generate thread key from URL if not provided
        if not thread_key:
            thread_key = normalize_url(url).rstrip("/").split("/")[-1]
            thread_key = re.sub(r"[^\w\-_.]", "_", thread_key)

        thread_dir = get_thread_dir(thread_key)

        # Check if we need to fetch pages
        html_files = [f for f in os.listdir(thread_dir) if f.endswith(".html")]

        if url and (refresh or not html_files):
            logger.info(f"Fetching forum pages for {url}")
            last_page = detect_last_page(url)
            fetch_forum_pages(url, last_page, thread_dir)

        # Preprocess the thread
        preprocess_thread(thread_dir, force=refresh)

        return jsonify(
            {
                "status": "success",
                "thread_key": thread_key,
                "message": f"Thread {thread_key} processed successfully",
            }
        )

    except Exception as e:
        logger.error(f"Error in preprocess route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_route():
    """Ask a question about a forum thread with intelligent query processing."""
    try:
        data = request.get_json(force=True) or {}
        prompt = data.get("prompt", "").strip()
        url = data.get("url", "").strip()
        existing_thread = data.get("existing_thread", "").strip()
        refresh = data.get("refresh", False)

        if not prompt:
            return "Prompt is required", 400

        if not url and not existing_thread:
            return "URL or existing_thread is required", 400

        # Determine thread key
        if existing_thread:
            thread_key = existing_thread
        else:
            thread_key = normalize_url(url).rstrip("/").split("/")[-1]
            thread_key = re.sub(r"[^\w\-_.]", "_", thread_key)

        thread_dir = get_thread_dir(thread_key)

        # Ensure thread is processed
        html_files = [f for f in os.listdir(thread_dir) if f.endswith(".html")]

        if url and (refresh or not html_files):
            logger.info(f"Fetching forum pages for {url}")
            last_page = detect_last_page(url)
            fetch_forum_pages(url, last_page, thread_dir)

        preprocess_thread(thread_dir, force=refresh)

        # Use intelligent query processing
        intent, posts = process_intelligent_query(prompt, thread_dir)

        logger.info(
            f"Query intent: {intent.query_type.value} (confidence: {intent.confidence:.2f})"
        )

        # If intelligent retrieval found specific posts, use them
        if posts:
            logger.info(f"Using {len(posts)} posts from intelligent retrieval")
        else:
            # Fallback to semantic search
            logger.info("Falling back to semantic search")

            # Use the refined semantic query if available
            search_query = intent.semantic_query or prompt
            posts = find_relevant_posts(search_query, thread_dir)

        # Build context from posts
        if posts:
            context_parts = []
            for i, post in enumerate(posts[:FINAL_TOP_K], 1):
                part = (
                    f"--- Post {i} by {post['author']} on {post['date']} (Page {post['page']}) ---\n"
                    f"URL: {post['url']}\n\n{post['content']}\n"
                )
                context_parts.append(part)
            context = "\n".join(context_parts)
            logger.info(f"Built context from {len(posts)} posts ({len(context)} chars)")
        else:
            context = "No relevant information found in the forum thread."
            logger.warning("No posts found for query")

        # Create enhanced system prompt based on intent
        system_prompt = create_enhanced_system_prompt(intent, context)
        system_prompt += (
            f"\n\nUser Question: {prompt}\n\nPlease provide a detailed answer:"
        )

        def generate_response():
            """Stream LLM response with enhanced context."""
            try:
                logger.info("Starting enhanced LLM response generation")
                response = requests.post(
                    OLLAMA_API_URL,
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": system_prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.2,  # Lower temperature for more precise answers
                            "top_p": 0.9,
                            "max_tokens": 2000,
                        },
                    },
                    stream=True,
                    timeout=120,
                )
                response.raise_for_status()

                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                logger.error(f"LLM streaming error: {e}")
                yield f"\n\n[Error: Failed to generate response - {str(e)}]\n"

        return Response(generate_response(), mimetype="text/plain")

    except Exception as e:
        logger.error(f"Error in enhanced ask route: {e}")
        return f"Error: {str(e)}", 500


@app.route("/delete_thread", methods=["POST"])
def delete_route():
    """Delete a thread and its data."""
    try:
        data = request.json or {}
        thread_key = data.get("thread_key", "").strip()

        if not thread_key:
            return jsonify({"error": "thread_key is required"}), 400

        # Sanitize thread key
        if any(c in thread_key for c in ("..", "/", "\\")):
            return jsonify({"error": "Invalid thread key"}), 400

        thread_dir = os.path.join(BASE_TMP_DIR, thread_key)

        if os.path.exists(thread_dir):
            shutil.rmtree(thread_dir)
            logger.info(f"Deleted thread: {thread_key}")
            return jsonify(
                {"status": "success", "message": f"Thread {thread_key} deleted"}
            )
        else:
            return jsonify({"error": "Thread not found"}), 404

    except Exception as e:
        logger.error(f"Error deleting thread: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "threads": len(list_threads()),
            "cache_size": len(load_cache()) if os.path.exists(CACHE_PATH) else 0,
        }
    )


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Ensure base directory exists
    os.makedirs(BASE_TMP_DIR, exist_ok=True)

    # Create log file
    log_file = os.path.join(BASE_TMP_DIR, "app.log")
    try:
        with open(log_file, "a") as f:
            f.write(
                f"\n--- Application started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
    except Exception as e:
        logger.warning(f"Could not write to log file: {e}")

    logger.info("Starting Forum Analysis Application")
    logger.info(f"Base directory: {BASE_TMP_DIR}")
    logger.info(f"Ollama API: {OLLAMA_API_URL}")
    logger.info(f"Embed API: {OLLAMA_EMBED_API_URL}")
    logger.info(f"Model: {OLLAMA_MODEL}")
    logger.info(f"Embed Model: {OLLAMA_EMBED_MODEL}")

    # Start the Flask app
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

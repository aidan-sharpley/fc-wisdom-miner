import hashlib
import html
import json
import logging
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import urljoin

import hnswlib
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, Response, render_template, request
from tqdm import tqdm

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-a-secure-random-secret")

# --- Configuration ---
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_API_URL = os.environ.get(
    "OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
BASE_TMP_DIR = os.environ.get("BASE_TMP_DIR", "tmp")
INDEX_META_NAME, HNSW_INDEX_NAME = "index_meta.pkl", "index_hnsw.bin"
METADATA_INDEX_NAME = "metadata_index.json"
CACHE_PATH = os.path.join(BASE_TMP_DIR, "embeddings_cache.pkl")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Utility Functions ---
def post_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else "https://" + url


def get_thread_dir(thread_key: str) -> str:
    thread_dir = os.path.join(BASE_TMP_DIR, thread_key)
    os.makedirs(thread_dir, exist_ok=True)
    return thread_dir


def list_threads() -> List[str]:
    if not os.path.exists(BASE_TMP_DIR):
        return []
    return sorted(
        [
            d
            for d in os.listdir(BASE_TMP_DIR)
            if os.path.isdir(os.path.join(BASE_TMP_DIR, d))
        ]
    )


def clean_post_content(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(
        r"(?:^|\n)\s*\w+\s+said:\s*\n.*?(?:Click to expand\.\.\.)?",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"Click to expand\.\.\.", "", text)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


# --- HTML Parsing ---
def extract_date(el) -> str:
    return (el.find("time") or {}).get("datetime", "unknown-date")


def extract_author(el) -> str:
    return el.get("data-author", "unknown-author")


def extract_content(el) -> str:
    for selector in ["div.message-userContent .bbWrapper", ".message-body .bbWrapper"]:
        content_div = el.select_one(selector)
        if content_div:
            return content_div.get_text(separator="\n").strip()
    return ""


def extract_post_url(el: BeautifulSoup, base: str) -> str:
    link = el.select_one("ul.message-attribution-opposite a[href]")
    if link and link["href"]:
        return urljoin(base, link["href"])
    if el.get("data-content", "").startswith("post-"):
        post_id = el["data-content"].split("-")[1]
        return urljoin(base, f"posts/{post_id}/")
    return f"{base}#unknown-post-id"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    link = soup.find("link", rel="canonical")
    if link and link.has_attr("href"):
        return re.sub(r"page-\d+/?$", "", link["href"]).rstrip("/") + "/"
    return "unknown-thread/"


# --- Web Scraping & File Handling ---
def detect_last_page(url: str) -> int:
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        page_jump = soup.select_one("input.js-pageJumpPage[max]")
        if page_jump:
            return int(page_jump["max"])
        last_link = soup.select_one(".pageNav-main > li:last-child a")
        if last_link and last_link.text.isdigit():
            return int(last_link.text)
        return 1
    except Exception as e:
        logger.error(f"Could not detect last page for {url}: {e}")
        return 1


def fetch_forum_pages(base_url: str, end: int, save_dir: str):
    base_url = re.sub(r"page-\d+/?$", "", base_url).rstrip("/")
    for i in range(1, end + 1):
        path = os.path.join(save_dir, f"page{i}.html")
        if not os.path.exists(path):
            logger.info(f"Fetching page {i}/{end}...")
            try:
                res = requests.get(f"{base_url}/page-{i}", timeout=10)
                res.raise_for_status()
                with open(path, "w", encoding="utf-8") as f:
                    f.write(res.text)
                time.sleep(0.5)
            except requests.RequestException as e:
                logger.error(f"Failed to fetch page {i}: {e}")
                break


# --- Caching & Embeddings ---
def load_cache() -> Dict[str, np.ndarray]:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            logger.warning(f"Cache file {CACHE_PATH} is corrupted. Starting fresh.")
    return {}


def save_cache(cache: Dict[str, np.ndarray]):
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_text(text: str) -> np.ndarray:
    res = requests.post(
        OLLAMA_EMBED_API_URL, json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
    )
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype="float32")


# --- Core Logic: Preprocessing & Search ---
def preprocess_thread(thread_dir: str, force: bool = False):
    paths = {
        p: os.path.join(thread_dir, p)
        for p in [INDEX_META_NAME, HNSW_INDEX_NAME, METADATA_INDEX_NAME, "posts"]
    }
    if not force and all(os.path.exists(v) for v in paths.values()):
        logger.info(f"Index exists for {thread_dir}, skipping preprocessing.")
        return

    logger.info(f"Starting preprocessing for {thread_dir}")
    if force and os.path.exists(paths["posts"]):
        shutil.rmtree(paths["posts"])
    os.makedirs(paths["posts"], exist_ok=True)

    html_files = sorted(
        [f for f in os.listdir(thread_dir) if f.endswith(".html")],
        key=lambda x: int(re.search(r"page(\d+)", x).group(1)),
    )
    raw_posts, metadata_list = [], []
    for fn in html_files:
        page_num = int(re.search(r"page(\d+)", fn).group(1))
        with open(os.path.join(thread_dir, fn), encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        base_url = extract_canonical_url(soup)
        for el in soup.select("article.message"):
            content = clean_post_content(extract_content(el))
            if content:
                post_data = {
                    "page": page_num,
                    "date": extract_date(el),
                    "author": extract_author(el),
                    "content": content,
                    "url": extract_post_url(el, base_url),
                }
                raw_posts.append(post_data)
                metadata_list.append(
                    {
                        "author": post_data["author"],
                        "date": post_data["date"],
                        "page": post_data["page"],
                    }
                )

    with open(paths[METADATA_INDEX_NAME], "w") as f:
        json.dump(metadata_list, f)

    cache = load_cache()
    to_embed = [
        p["content"] for p in raw_posts if post_hash(p["content"]) not in cache or force
    ]
    logger.info(
        f"{len(raw_posts) - len(to_embed)} posts in cache. Embedding {len(to_embed)} new posts."
    )

    if to_embed:
        with ThreadPoolExecutor() as ex:
            future_map = {
                ex.submit(embed_text, content): content for content in to_embed
            }
            for future in tqdm(
                as_completed(future_map), total=len(to_embed), desc="Embedding"
            ):
                try:
                    cache[post_hash(future_map[future])] = future.result()
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
        save_cache(cache)

    final_embeddings = []
    for i, post in enumerate(raw_posts):
        h = post_hash(post["content"])
        if h in cache:
            final_embeddings.append(cache[h])
            with open(os.path.join(paths["posts"], f"{i}.json"), "w") as f:
                json.dump(post, f)

    if not final_embeddings:
        return logger.error("No embeddings found, aborting index.")

    dim, count = final_embeddings[0].shape[0], len(final_embeddings)
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=count, ef_construction=200, M=32)
    idx.add_items(np.vstack(final_embeddings))
    idx.save_index(paths[HNSW_INDEX_NAME])
    with open(paths[INDEX_META_NAME], "wb") as f:
        pickle.dump({"dim": dim, "count": count}, f)
    logger.info(f"HNSW index with {count} items saved.")


def generate_hypothetical_answer(query: str) -> str:
    prompt = f"Write a detailed, hypothetical answer to this question to help find relevant documents.\n\nQuestion: {query}"
    try:
        res = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=15,
        )
        res.raise_for_status()
        return res.json().get("response", query)
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return query


def batch_llm_score_posts(query: str, posts: List[Dict]) -> List[tuple[int, Dict]]:
    lines = [
        f"Query: {query}\nRate relevance of each post from 0-10. Respond ONLY with a numbered list of scores (e.g., '1. 7').\n"
    ]
    for i, post in enumerate(posts, 1):
        lines.append(f"{i}. Post by {post['author']}: {post['content'][:400]}...")
    try:
        res = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": "\n".join(lines), "stream": False},
            timeout=30,
        )
        res.raise_for_status()
        text = res.json().get("response", "")
        scores = [int(m.group(1)) for m in re.finditer(r"\d+\.\s*(\d+)", text)]
        return (
            list(zip(scores, posts))
            if len(scores) == len(posts)
            else [(0, p) for p in posts]
        )
    except Exception as e:
        logger.error(f"LLM reranking failed: {e}")
        return [(0, p) for p in posts]


def find_posts_by_metadata(
    thread_dir: str,
    author: Optional[str] = None,
    page: Optional[int] = None,
    date: Optional[str] = None,
) -> List[Dict]:
    """Efficiently finds posts by filtering the metadata index."""
    posts_dir = os.path.join(thread_dir, "posts")
    metadata_path = os.path.join(thread_dir, METADATA_INDEX_NAME)
    if not os.path.exists(metadata_path):
        return []

    with open(metadata_path, "r") as f:
        metadata_list = json.load(f)

    matching_indices = []
    for i, meta in enumerate(metadata_list):
        if author and author.lower() in meta.get("author", "").lower():
            matching_indices.append(i)
        elif page and page == meta.get("page"):
            matching_indices.append(i)
        elif date and date.lower() in meta.get("date", "").lower():
            matching_indices.append(i)

    results = []
    # To avoid overwhelming the context, limit results for broad queries
    for i in matching_indices[:15]:
        try:
            with open(os.path.join(posts_dir, f"{i}.json"), "r") as f:
                results.append(json.load(f))
        except FileNotFoundError:
            continue
    logger.info(f"Found {len(results)} posts via metadata filter.")
    return results


def find_relevant_posts(query: str, thread_dir: str, top_k: int = 7) -> List[Dict]:
    """Finds relevant posts using HyDE, semantic search, and reranking."""
    paths = {
        p: os.path.join(thread_dir, p)
        for p in [INDEX_META_NAME, HNSW_INDEX_NAME, "posts"]
    }
    if not all(os.path.exists(v) for v in paths.values()):
        return []

    with open(paths[INDEX_META_NAME], "rb") as f:
        meta = pickle.load(f)
    idx = hnswlib.Index(space="cosine", dim=meta["dim"])
    idx.load_index(paths[HNSW_INDEX_NAME])

    hyde_query = generate_hypothetical_answer(query)
    query_emb = embed_text(hyde_query)
    labels, _ = idx.knn_query(query_emb, k=25)

    candidates = [
        json.load(open(os.path.join(paths["posts"], f"{i}.json")))
        for i in labels[0]
        if os.path.exists(os.path.join(paths["posts"], f"{i}.json"))
    ]
    scored_posts = sorted(
        batch_llm_score_posts(query, candidates), key=lambda x: x[0], reverse=True
    )

    final_posts = [post for score, post in scored_posts[:top_k]]
    logger.info(
        f"Retrieved {len(candidates)} candidates, returning top {len(final_posts)} after reranking."
    )
    return final_posts


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html", threads=list_threads())


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    prompt, url, existing, refresh = (
        data.get("prompt", "").strip(),
        data.get("url", "").strip(),
        data.get("existing_thread", "").strip(),
        data.get("refresh", False),
    )

    if not prompt or not (url or existing):
        return "Prompt and URL/existing thread required.", 400

    thread_key = existing or normalize_url(url).rstrip("/").split("/")[-1].split(".")[0]
    thread_dir = get_thread_dir(thread_key)

    if url and (
        not any(f.endswith(".html") for f in os.listdir(thread_dir))
        or (refresh and not existing)
    ):
        logger.info(f"Fetching pages for '{thread_key}'.")
        fetch_forum_pages(url, detect_last_page(url), thread_dir)

    preprocess_thread(thread_dir, force=refresh)

    # --- Hybrid Search Logic ---
    posts = []
    p_lower = prompt.lower()

    author_match = re.search(r"posts by\s+([a-zA-Z0-9_-]+)", p_lower)
    page_match = re.search(r"on page\s+(\d+)", p_lower)
    date_match = re.search(
        r"from\s+([a-zA-Z]+\s+\d{1,2},?\s+\d{4})", p_lower
    )  # Matches "May 4, 2024" or "May 4 2024"

    if any(s in p_lower for s in ["first post", "earliest post"]):
        logger.info("Intent: Get first post.")
        try:
            posts = [json.load(open(os.path.join(thread_dir, "posts", "0.json")))]
        except FileNotFoundError:
            logger.error("Could not find first post.")
    elif any(s in p_lower for s in ["last post", "newest post"]):
        logger.info("Intent: Get last post.")
        try:
            with open(os.path.join(thread_dir, INDEX_META_NAME), "rb") as f:
                count = pickle.load(f).get("count", 0)
            if count > 0:
                posts = [
                    json.load(
                        open(os.path.join(thread_dir, "posts", f"{count - 1}.json"))
                    )
                ]
        except FileNotFoundError:
            logger.error("Could not find last post.")
    elif author_match:
        logger.info(f"Intent: Get posts by author '{author_match.group(1)}'.")
        posts = find_posts_by_metadata(thread_dir, author=author_match.group(1))
    elif page_match:
        logger.info(f"Intent: Get posts on page {page_match.group(1)}.")
        posts = find_posts_by_metadata(thread_dir, page=int(page_match.group(1)))
    elif date_match:
        logger.info(f"Intent: Get posts on date '{date_match.group(1)}'.")
        posts = find_posts_by_metadata(thread_dir, date=date_match.group(1))

    if not posts:
        posts = find_relevant_posts(prompt, thread_dir)  # Fallback to semantic search

    context = (
        "\n\n".join(
            f"--- Post by {p['author']} on {p['date']} (Page {p['page']}) ---\nURL: {p['url']}\n\n{p['content']}"
            for p in posts
        )
        if posts
        else "No relevant information found."
    )
    system = f"You are an expert forum analyst. Use *only* the provided context below to answer the question.\n\n---CONTEXT---\n{context}\n---END CONTEXT---\n\nQuestion: {prompt}"

    def stream_response():
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": system, "stream": True}
            with requests.post(OLLAMA_API_URL, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        yield json.loads(line).get("response", "")
        except Exception as e:
            logger.error(f"Error streaming from LLM: {e}")
            yield "Error communicating with the language model."

    return Response(stream_response(), mimetype="text/plain")


@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    key = request.get_json().get("thread_key", "").strip()
    if not key or any(c in key for c in ("..", "/")):
        return "Invalid thread key", 400
    td = os.path.join(BASE_TMP_DIR, key)
    if os.path.exists(td):
        shutil.rmtree(td)
        return f"Deleted {key}", 200
    return "Not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)

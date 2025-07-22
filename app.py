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
from typing import Dict, List
from urllib.parse import urljoin

import hnswlib
import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, Response, render_template, request
from tqdm import tqdm

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-a-secure-random-secret")

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_API_URL = os.environ.get(
    "OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")

BASE_TMP_DIR = os.environ.get("BASE_TMP_DIR", "tmp")
INDEX_META_NAME = "index_meta.pkl"
HNSW_INDEX_NAME = "index_hnsw.bin"
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
        d
        for d in os.listdir(BASE_TMP_DIR)
        if os.path.isdir(os.path.join(BASE_TMP_DIR, d))
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


# --- HTML Parsing Functions ---


def extract_date(post_element) -> str:
    time_tag = post_element.find("time")
    return (
        time_tag["datetime"]
        if time_tag and time_tag.has_attr("datetime")
        else "unknown-date"
    )


def extract_author(post_element) -> str:
    return post_element.get("data-author", "unknown-author")


def extract_content(post_element) -> str:
    for selector in ["div.message-userContent .bbWrapper", ".message-body .bbWrapper"]:
        content_div = post_element.select_one(selector)
        if content_div:
            return content_div.get_text(separator="\n").strip()
    return ""


def extract_post_url(post_element: BeautifulSoup, canonical_base: str) -> str:
    link = post_element.select_one("ul.message-attribution-opposite a[href]")
    if link and link["href"]:
        return urljoin(canonical_base, link["href"])
    if post_element.get("data-content", "").startswith("post-"):
        post_id = post_element["data-content"].split("-")[1]
        return urljoin(canonical_base, f"posts/{post_id}/")
    return f"{canonical_base}#unknown-post-id"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    link = soup.find("link", rel="canonical")
    if link and link.has_attr("href"):
        url = link["href"]
        return re.sub(r"page-\d+/?$", "", url).rstrip("/") + "/"
    return "unknown-thread/"


# --- Web Scraping and File Handling ---


def detect_last_page(base_url: str) -> int:
    try:
        res = requests.get(base_url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        page_jump = soup.select_one("input.js-pageJumpPage[max]")
        if page_jump:
            return int(page_jump["max"])
        last_page_link = soup.select_one(".pageNav-main > li:last-child a")
        if last_page_link and last_page_link.text.isdigit():
            return int(last_page_link.text)
        return 1
    except Exception as e:
        logger.error(f"Could not detect last page: {e}")
        return 1


def fetch_forum_pages(base_url: str, start: int, end: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    base_url = re.sub(r"page-\d+/?$", "", base_url).rstrip("/")
    for i in range(start, end + 1):
        page_url = f"{base_url}/page-{i}"
        output_path = os.path.join(save_dir, f"page{i}.html")
        if not os.path.exists(output_path):
            logger.info(f"Fetching page {i}...")
            try:
                res = requests.get(page_url)
                res.raise_for_status()
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(res.text)
                time.sleep(0.5)
            except requests.RequestException as e:
                logger.error(f"Failed to fetch {page_url}: {e}")
                break


# --- Caching and Embeddings ---


def load_cache() -> Dict[str, np.ndarray]:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            logger.warning(f"Cache file {CACHE_PATH} is corrupted. Starting fresh.")
    return {}


def save_cache(cache: Dict[str, np.ndarray]):
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_text(text: str) -> np.ndarray:
    res = requests.post(
        OLLAMA_EMBED_API_URL, json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
    )
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype="float32")


# --- Core Preprocessing and Search Logic ---


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    meta_path, hnsw_path = (
        os.path.join(thread_dir, INDEX_META_NAME),
        os.path.join(thread_dir, HNSW_INDEX_NAME),
    )
    posts_dir = os.path.join(thread_dir, "posts")

    if not force and all(os.path.exists(p) for p in [meta_path, hnsw_path, posts_dir]):
        logger.info(f"Index exists for {thread_dir}, skipping preprocessing.")
        return

    logger.info(f"Starting preprocessing for thread at {thread_dir}")
    if force and os.path.exists(posts_dir):
        shutil.rmtree(posts_dir)
    os.makedirs(posts_dir, exist_ok=True)

    html_files = sorted(
        [f for f in os.listdir(thread_dir) if f.endswith(".html")],
        key=lambda x: int(re.search(r"page(\d+)", x).group(1)),
    )

    raw_posts = []
    for fn in html_files:
        page_num = int(re.search(r"page(\d+)", fn).group(1))
        with open(os.path.join(thread_dir, fn), encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        base_url = extract_canonical_url(soup)
        for el in soup.select("article.message"):
            content = clean_post_content(extract_content(el))
            if content:
                raw_posts.append(
                    {
                        "page": page_num,
                        "date": extract_date(el),
                        "author": extract_author(el),
                        "content": content,
                        "url": extract_post_url(el, base_url),
                    }
                )

    logger.info(f"Found {len(raw_posts)} posts to process.")
    cache = load_cache()

    to_embed_contents = [
        post["content"] for post in raw_posts if post_hash(post["content"]) not in cache
    ]
    logger.info(
        f"{len(raw_posts) - len(to_embed_contents)} posts found in cache. Embedding {len(to_embed_contents)} new posts."
    )

    if to_embed_contents:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_content = {
                executor.submit(embed_text, content): content
                for content in to_embed_contents
            }
            for future in tqdm(
                as_completed(future_to_content),
                total=len(to_embed_contents),
                desc="Embedding",
            ):
                try:
                    embedding = future.result()
                    content = future_to_content[future]
                    cache[post_hash(content)] = embedding
                except Exception as e:
                    logger.error(f"Embedding failed for a post: {e}")
        save_cache(cache)

    final_embeddings = []
    for i, post in enumerate(raw_posts):
        h = post_hash(post["content"])
        if h in cache:
            final_embeddings.append(cache[h])
            with open(os.path.join(posts_dir, f"{i}.json"), "w") as f:
                json.dump(post, f)

    if not final_embeddings:
        logger.error("No valid embeddings were generated. Aborting index creation.")
        return

    dim, num_elements = final_embeddings[0].shape[0], len(final_embeddings)
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=num_elements, ef_construction=200, M=32)
    idx.add_items(np.vstack(final_embeddings))
    idx.save_index(hnsw_path)

    with open(meta_path, "wb") as f:
        pickle.dump({"dim": dim, "count": num_elements}, f)
    logger.info(f"HNSW index with {num_elements} items saved successfully.")


def batch_llm_score_posts(query: str, posts: List[Dict]) -> List[tuple[int, Dict]]:
    prompt_lines = [
        f"Query: {query}\nRate the relevance of each post below from 0 to 10. Respond ONLY with a numbered list of scores (e.g., '1. 7').\n"
    ]
    for i, post in enumerate(posts, 1):
        prompt_lines.append(
            f"{i}. Post by {post['author']}: {post['content'][:400]}..."
        )

    try:
        res = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": "\n".join(prompt_lines),
                "stream": False,
            },
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


def generate_hypothetical_answer(query: str) -> str:
    prompt = f"Write a detailed, hypothetical answer to the user's question. This will be used to find relevant documents.\n\nQuestion: {query}"
    try:
        res = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        res.raise_for_status()
        return res.json().get("response", query)
    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return query


def find_relevant_posts(query: str, thread_dir: str, top_k: int = 5) -> List[Dict]:
    meta_path, hnsw_path, posts_dir = [
        os.path.join(thread_dir, name)
        for name in [INDEX_META_NAME, HNSW_INDEX_NAME, "posts"]
    ]
    if not all(os.path.exists(p) for p in [meta_path, hnsw_path, posts_dir]):
        return []

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    idx = hnswlib.Index(space="cosine", dim=meta["dim"])
    idx.load_index(hnsw_path)

    hyde_query = generate_hypothetical_answer(query)
    query_emb = embed_text(hyde_query)
    candidate_labels, _ = idx.knn_query(query_emb, k=25)

    candidates = []
    for i in candidate_labels[0]:
        try:
            with open(os.path.join(posts_dir, f"{i}.json"), "r") as f:
                candidates.append(json.load(f))
        except FileNotFoundError:
            continue

    scored_posts = batch_llm_score_posts(query, candidates)
    scored_posts.sort(key=lambda x: x[0], reverse=True)

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
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    url = data.get("url", "").strip()
    existing = data.get("existing_thread", "").strip()
    refresh = data.get("refresh", False)

    if not prompt or (not url and not existing):
        return "Prompt and URL/existing thread are required.", 400

    thread_key = existing or normalize_url(url).rstrip("/").split("/")[-1].split(".")[0]
    logger.info(f"Processing request for thread: '{thread_key}'")
    thread_dir = get_thread_dir(thread_key)

    has_html = any(f.endswith(".html") for f in os.listdir(thread_dir))
    if url and (not has_html or refresh):
        logger.info(f"Fetching pages for '{thread_key}'. Refresh: {refresh}")
        last_page = detect_last_page(url)
        fetch_forum_pages(url, 1, last_page, save_dir=thread_dir)

    preprocess_thread(thread_dir, force=refresh)
    posts = find_relevant_posts(prompt, thread_dir)

    context = (
        "\n\n".join(
            f"--- Post by {p['author']} on {p['date']} (Page {p['page']}) ---\nURL: {p['url']}\n\n{p['content']}"
            for p in posts
        )
        if posts
        else "No relevant information was found in the thread to answer the question."
    )

    system = f"""You are an expert forum analyst. Use *only* the provided context below to answer the question. The context consists of several posts from a forum thread. If the answer cannot be found in the context, say so.

---CONTEXT---
{context}
---END CONTEXT---

Question: {prompt}
"""

    def stream_response():
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": system, "stream": True}
            with requests.post(OLLAMA_API_URL, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
        except Exception as e:
            logger.error(f"Error streaming from LLM: {e}")
            yield "Error communicating with the language model."

    return Response(stream_response(), mimetype="text/plain")


@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    data = request.get_json(force=True)
    key = data.get("thread_key", "").strip()
    if not key or any(c in key for c in ("..", "/")):
        return "Invalid thread key", 400
    td = os.path.join(BASE_TMP_DIR, key)
    if os.path.exists(td):
        logger.info(f"Deleting thread directory: {td}")
        shutil.rmtree(td)
        return f"Deleted {key}", 200
    return "Not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)

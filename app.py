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
# Switch to the GPU‑accelerated embedding model by default:
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


def extract_date(post_element) -> str:
    time_tag = post_element.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        return time_tag["datetime"]
    return "unknown-date"


def extract_content(post_element) -> str:
    # Pulls post content from XenForo bbWrapper
    content_div = post_element.select_one(".message-main .bbWrapper")
    return content_div.get_text(separator="\n").strip() if content_div else ""


def extract_post_url(post_element: BeautifulSoup, canonical_base: str) -> str:
    """Build a full permalink from relative href or fallback to post ID."""
    permalink_element = post_element.select_one(".message-attribution-main a")
    if permalink_element and permalink_element.get("href"):
        # The urljoin function correctly handles joining the base URL with a relative path
        return urljoin(canonical_base, permalink_element["href"])

    # Fallback: construct from the 'data-content' attribute which is like 'post-12345'
    if post_element.get("data-content", "").startswith("post-"):
        post_id = post_element["data-content"].split("-")[1]
        # Reconstruct a likely permalink structure.
        # e.g., base: '.../threads/slug/' + 'post-123/' -> '.../threads/slug/post-123/'
        logger.debug(f"Using fallback URL construction for post ID {post_id}")
        return urljoin(canonical_base, f"post-{post_id}/")

    logger.warning(
        f"Could not determine a permalink for post by user {post_element.get('data-author')}. Using a non-specific URL."
    )
    return f"{canonical_base}unknown-post"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    """Grabs the base thread URL from the <link rel='canonical'> tag."""
    canonical_link = soup.select_one("head link[rel='canonical']")
    return (
        canonical_link["href"].rstrip("/") + "/"
        if canonical_link
        else "unknown-thread/"
    )


def detect_last_page(base_url: str) -> int:
    res = requests.get(base_url)
    if res.status_code != 200:
        logger.error(f"Error: Received status code {res.status_code}")
        return 1

    soup = BeautifulSoup(res.text, "html.parser")
    logger.info("Base page pulled successfully.")

    last_page = 1  # Initialize with 1

    # First, try to find the 'max' attribute in the page jump input, if available.
    # This is often the most direct way to get the true last page.
    page_jump_input = soup.select_one('.js-pageJumpPage[type="number"]')
    if page_jump_input and "max" in page_jump_input.attrs:
        try:
            max_page_from_input = int(page_jump_input["max"])
            if max_page_from_input > last_page:
                last_page = max_page_from_input
                logger.info(f"Detected last page from page jump input: {last_page}")
                return last_page  # If found, this is highly reliable, so return immediately.
        except ValueError:
            pass  # Ignore if max is not a valid number

    # Fallback: Iterate through all page number links and find the highest one.
    # This handles cases where there's no explicit 'last' link or max attribute.
    page_links = soup.select(".pageNav-page a")  # Select <a> tags within .pageNav-page
    for link in page_links:
        try:
            # Try to get number from href (e.g., page-227)
            href = link.get("href")
            if href:
                match = re.search(r"page-(\d+)", href)
                if match:
                    page_num_from_href = int(match.group(1))
                    if page_num_from_href > last_page:
                        last_page = page_num_from_href

            # Also try to get number from text content (e.g., 227)
            link_text = link.text.strip()
            if link_text.isdigit():
                page_num_from_text = int(link_text)
                if page_num_from_text > last_page:
                    last_page = page_num_from_text
        except ValueError:
            continue  # Ignore links that don't contain valid numbers

    logger.info(f"Detected last page: {last_page}")
    return last_page


def fetch_forum_pages(base_url: str, start: int, end: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for i in range(start, end + 1):
        page_url = base_url if i == 1 else f"{base_url}page-{i}"
        output_path = os.path.join(save_dir, f"page{i}.html")

        if os.path.exists(output_path):
            continue

        logger.info(f"Fetching page {i}...")
        res = requests.get(page_url)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(res.text)

        time.sleep(0.5)


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


def load_cache() -> Dict[str, np.ndarray]:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: Dict[str, np.ndarray]):
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text using the GPU‑accelerated Ollama model.
    """
    res = requests.post(
        OLLAMA_EMBED_API_URL,
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
    )
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype="float32")


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    hnsw_path = os.path.join(thread_dir, HNSW_INDEX_NAME)

    if not force and os.path.exists(meta_path) and os.path.exists(hnsw_path):
        logger.info(f"[Preprocess] Index already exists for {thread_dir}, skipping.")
        return

    logger.info(f"[Preprocess] Starting for thread at {thread_dir}")

    # 1) Collect raw posts
    raw_posts = []
    for fn in sorted(os.listdir(thread_dir)):
        match = re.search(r"page[-_]?(\d+)\.html", fn)
        if not match:
            logger.debug(f"[Preprocess] Skipping file {fn}")
            continue
        page = int(match.group(1))
        with open(os.path.join(thread_dir, fn), encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        base_thread_url = extract_canonical_url(soup)

        for el in soup.select("article.message"):
            content = extract_content(el).strip()
            if content:
                raw_posts.append(
                    {
                        "page": page,
                        "date": extract_date(el),
                        "content": clean_post_content(content),
                        "url": extract_post_url(el, base_thread_url),
                    }
                )
            else:
                logger.warning(f"Empty content on page {page}, skipping post.")

    logger.info(f"[Preprocess] Found {len(raw_posts)} posts to embed.")

    # 2) Load cache and prepare lists
    cache = load_cache()

    def post_hash(post):
        return hashlib.sha256(post["content"].encode("utf-8")).hexdigest()

    embeddings, posts, to_embed = [], [], []
    for post in raw_posts:
        h = post_hash(post)
        if h in cache and not force:
            logger.debug(f"Cache hit for post hash {h[:8]}...")
            embeddings.append(cache[h])
            posts.append(post)
        else:
            to_embed.append((h, post))

    logger.info(
        f"[Preprocess] {len(posts)} posts loaded from cache; {len(to_embed)} new posts to embed."
    )

    # 3) Parallel embed the uncached posts
    if to_embed:
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = {
                exe.submit(embed_text, p["content"]): (h, p) for h, p in to_embed
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Embedding posts"
            ):
                h, p = futures[future]
                try:
                    emb = future.result()
                    cache[h] = emb
                    embeddings.append(emb)
                    posts.append(p)
                except Exception as e:
                    logger.error(f"[Preprocess] Embed error for page {p['page']}: {e}")

    # 4) Persist cache
    save_cache(cache)
    logger.info(f"[Preprocess] Cache saved ({len(cache)} total embeddings).")

    # Remove any zero‑length embeddings (and corresponding posts)
    filtered = [(e, p) for e, p in zip(embeddings, posts) if e.ndim == 1 and e.size > 0]
    if not filtered:
        logger.warning(
            "[Preprocess] No valid embeddings found, aborting index creation."
        )
        return

    if len(filtered) < len(embeddings):
        logger.warning(
            f"Filtered out {len(embeddings) - len(filtered)} posts with invalid embeddings."
        )

    embeddings, posts = zip(*filtered)

    # 5) Build HNSW index
    dim = embeddings[0].shape[0]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=len(embeddings), ef_construction=200, M=32)
    idx.add_items(np.vstack(embeddings))
    idx.set_ef(50)

    idx.save_index(hnsw_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"posts": posts, "dim": dim}, f)
    logger.info(f"[Preprocess] HNSW index ({len(embeddings)} items, dim={dim}) saved.")


def find_relevant_posts(query: str, thread_dir: str, top_k: int = 5) -> List[Dict]:
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    hnsw_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    if not os.path.exists(meta_path) or not os.path.exists(hnsw_path):
        logger.warning(f"[Search] Missing index/meta for {thread_dir}, cannot search.")
        return []

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    posts, dim = meta["posts"], meta["dim"]

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(hnsw_path)
    idx.set_ef(50)

    try:
        q_emb = embed_text(query)
    except Exception as e:
        logger.error(f"[Search] Query embed failed: {e}")
        return []

    labels, _ = idx.knn_query(q_emb, k=top_k)
    results = [posts[i] for i in labels[0] if i < len(posts)]
    logger.info(f"[Search] Returning {len(results)} relevant posts for query.")
    return results


@app.route("/")
def index():
    return render_template("index.html", threads=list_threads())


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    existing = data.get("existing_thread", "").strip()
    url = data.get("url", "").strip()
    prompt = data.get("prompt", "").strip()
    refresh = data.get("refresh", False)

    if not prompt or (not existing and not url):
        return "Prompt and thread key or URL required", 400

    thread_key = existing or normalize_url(url).rstrip("/").split("/")[-1]
    logger.info(f"Processing request for thread_key: '{thread_key}'")
    thread_dir = get_thread_dir(thread_key)

    htmls = [f for f in os.listdir(thread_dir) if f.endswith(".html")]
    if url and (not htmls or refresh):
        logger.info(
            f"Fetching new HTML pages for '{thread_key}'. Refresh flag: {refresh}"
        )
        last = detect_last_page(url)
        fetch_forum_pages(url, 1, last, save_dir=thread_dir)
    else:
        logger.info(f"Using existing HTML pages for '{thread_key}'.")

    preprocess_thread(thread_dir, force=refresh)

    posts = find_relevant_posts(prompt, thread_dir)
    if not posts:
        context = (
            "No relevant information was found in the thread to answer the question."
        )
    else:
        context = "\n\n---\n\n".join(
            f"URL: {p['url']}\n[Page {p['page']}] {p['date']}:\n{p['content']}"
            for p in posts
        )

    system = f"""You are an expert forum analyst. Use *only* the provided context below to answer the question. The context consists of several posts from a forum thread. If the answer cannot be found in the context, say 'I do not know.'

---CONTEXT---
{context}
---END CONTEXT---

Question: {prompt}
"""
    payload = {"model": OLLAMA_MODEL, "prompt": system, "stream": True}

    def stream_resp():
        try:
            with requests.post(OLLAMA_API_URL, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
        except Exception as e:
            logger.error(f"Error streaming from OLLAMA: {e}")
            yield f"[Error: Could not connect to the language model: {e}]"
        yield "\n"

    return Response(stream_resp(), content_type="text/plain")


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
    logger.warning(f"Attempted to delete non-existent thread: {key}")
    return "Not found", 404


if __name__ == "__main__":
    logger.info("Starting Forum Wisdom Miner…")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)

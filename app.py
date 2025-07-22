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


def post_hash(post):
    return hashlib.sha256(post["content"].encode("utf-8")).hexdigest()


def extract_date(post_element) -> str:
    time_tag = post_element.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        return time_tag["datetime"]
    return "unknown-date"


def extract_author(post_element) -> str:
    return post_element.get("data-author", "unknown-author")


def extract_content(post_element) -> str:
    for selector in [
        "div.message-userContent .bbWrapper",
        "div.bbWrapper",
        ".message-content .bbWrapper",
        ".message-body .bbWrapper",
    ]:
        content_div = post_element.select_one(selector)
        if content_div:
            return content_div.get_text(separator="\n").strip()
    return ""


def extract_post_url(post_element: BeautifulSoup, canonical_base: str) -> str:
    """Returns the full permalink to the post."""
    # Try permalink link
    link = post_element.select_one("ul.message-attribution-opposite a[href]")
    if link:
        return urljoin(canonical_base, link["href"])

    # Fallback: use data-content="post-XXXXX"
    if post_element.get("data-content", "").startswith("post-"):
        post_id = post_element["data-content"].split("-")[1]
        return urljoin(canonical_base, f"posts/{post_id}/")

    return canonical_base + "#unknown"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    """Extracts the canonical base thread URL."""
    link = soup.find("link", rel="canonical")
    if link and link.has_attr("href"):
        url = link["href"]
        return re.sub(r"page-\d+/?$", "", url).rstrip("/") + "/"
    return "unknown-thread/"


def detect_last_page(base_url: str) -> int:
    res = requests.get(base_url)
    if res.status_code != 200:
        logger.error(f"Error: Received status code {res.status_code}")
        return 1

    soup = BeautifulSoup(res.text, "html.parser")
    logger.info("Base page pulled successfully.")

    last_page = 1

    page_jump_input = soup.select_one('.js-pageJumpPage[type="number"]')
    if page_jump_input and "max" in page_jump_input.attrs:
        try:
            max_page_from_input = int(page_jump_input["max"])
            if max_page_from_input > last_page:
                last_page = max_page_from_input
                logger.info(f"Detected last page from page jump input: {last_page}")
                return last_page
        except ValueError:
            pass

    page_links = soup.select(".pageNav-page a")
    for link in page_links:
        try:
            href = link.get("href")
            if href:
                match = re.search(r"page-(\d+)", href)
                if match:
                    page_num_from_href = int(match.group(1))
                    if page_num_from_href > last_page:
                        last_page = page_num_from_href

            link_text = link.text.strip()
            if link_text.isdigit():
                page_num_from_text = int(link_text)
                if page_num_from_text > last_page:
                    last_page = page_num_from_text
        except ValueError:
            continue

    logger.info(f"Detected last page: {last_page}")
    return last_page


def fetch_forum_pages(base_url: str, start: int, end: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    base_url = re.sub(r"page-\d+/?$", "", base_url).rstrip("/")

    for i in range(start, end + 1):
        page_url = f"{base_url}/page-{i}"
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
        try:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            logger.warning(
                f"Cache file at {CACHE_PATH} is corrupted. Starting a new cache."
            )
            return {}
    return {}


def save_cache(cache: Dict[str, np.ndarray]):
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_text(text: str) -> np.ndarray:
    res = requests.post(
        OLLAMA_EMBED_API_URL,
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
    )
    res.raise_for_status()
    return np.array(res.json()["embedding"], dtype="float32")


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    hnsw_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")

    if not force and os.path.exists(meta_path) and os.path.exists(hnsw_path):
        logger.info(f"[Preprocess] Index exists for {thread_dir}, skipping.")
        return

    logger.info(f"[Preprocess] Starting for thread at {thread_dir}")
    if os.path.exists(posts_dir) and force:
        shutil.rmtree(posts_dir)
    os.makedirs(posts_dir, exist_ok=True)

    # CORRECTED LOGIC: First, filter for HTML files, then sort them.
    all_files_in_dir = os.listdir(thread_dir)
    html_files = [f for f in all_files_in_dir if f.endswith(".html")]
    # Use a more specific regex to safely extract the page number for sorting
    sorted_html_files = sorted(
        html_files, key=lambda x: int(re.search(r"page(\d+)\.html", x).group(1))
    )

    raw_posts = []
    # Now, loop through the correctly sorted list of HTML files
    for fn in sorted_html_files:
        page = int(re.search(r"page(\d+)\.html", fn).group(1))
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
                        "author": extract_author(el),
                        "content": clean_post_content(content),
                        "url": extract_post_url(el, base_thread_url),
                    }
                )
            else:
                logger.warning(f"Empty or unparseable post at page {page}, skipping.")

    logger.info(f"[Preprocess] Found {len(raw_posts)} posts to embed.")
    cache = load_cache()

    embeddings, to_embed, post_ids = [], [], []
    for i, post in enumerate(raw_posts):
        h = post_hash(post)
        with open(os.path.join(posts_dir, f"{i}.json"), "w") as f:
            json.dump(post, f)

        if h in cache and not force:
            embeddings.append(cache[h])
        else:
            to_embed.append((h, post["content"]))
            post_ids.append(i)

    logger.info(
        f"[Preprocess] {len(embeddings)} posts loaded from cache; {len(to_embed)} new posts to embed."
    )

    if to_embed:
        embed_map = {}
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = {
                exe.submit(embed_text, p_content): h for h, p_content in to_embed
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Embedding posts"
            ):
                h = futures[future]
                try:
                    emb = future.result()
                    cache[h] = emb
                    embed_map[h] = emb
                except Exception as e:
                    logger.error(f"[Preprocess] Embed error: {e}")

        temp_embeddings = [None] * len(raw_posts)
        for i in range(len(raw_posts)):
            if i in post_ids:
                h = to_embed[post_ids.index(i)][0]
                if h in embed_map:
                    temp_embeddings[i] = embed_map[h]
                else:
                    logger.warning(
                        f"[Preprocess] No embedding returned for post index {i}"
                    )
            else:
                h = post_hash(raw_posts[i])
                if h in cache:
                    temp_embeddings[i] = cache[h]
                else:
                    logger.warning(
                        f"[Preprocess] No cached embedding for post index {i}"
                    )

        embeddings = [e for e in temp_embeddings if e is not None]

    save_cache(cache)
    logger.info(f"[Preprocess] Cache saved ({len(cache)} total embeddings).")

    if not embeddings or not all(
        isinstance(e, np.ndarray) and e.ndim == 1 and e.size > 0 for e in embeddings
    ):
        logger.warning(
            f"[Preprocess] Invalid embeddings found. Total valid: {sum(e is not None and e.size > 0 for e in embeddings)}"
        )
        return

    dim = embeddings[0].shape[0]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=len(embeddings), ef_construction=200, M=32)
    idx.add_items(np.vstack(embeddings))
    idx.set_ef(50)

    idx.save_index(hnsw_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"dim": dim, "count": len(embeddings)}, f)
    logger.info(f"[Preprocess] HNSW index ({len(embeddings)} items, dim={dim}) saved.")


def find_relevant_posts(query: str, thread_dir: str, top_k: int = 5) -> List[Dict]:
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    hnsw_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")

    if not all(os.path.exists(p) for p in [meta_path, hnsw_path, posts_dir]):
        logger.warning(f"[Search] Missing index/meta for {thread_dir}, cannot search.")
        return []

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    dim = meta["dim"]

    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(hnsw_path)
    idx.set_ef(50)

    try:
        q_emb = embed_text(query)
    except Exception as e:
        logger.error(f"[Search] Query embed failed: {e}")
        return []

    labels, _ = idx.knn_query(q_emb, k=top_k)
    results = []
    for i in labels[0]:
        post_path = os.path.join(posts_dir, f"{i}.json")
        try:
            with open(post_path, "r") as f:
                results.append(json.load(f))
        except FileNotFoundError:
            logger.warning(f"Could not find post metadata file for index {i}")

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

    thread_key = existing or normalize_url(url).rstrip("/").split("/")[-1].split(".")[0]
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
        context = "\n\n".join(
            f"--- Post by {p['author']} on {p['date']} (Page {p['page']}) ---\nURL: {p['url']}\n\n{p['content']}"
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

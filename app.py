import hashlib
import html
import json
import logging
import os
import pickle
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import hnswlib
import numpy as np
import requests
from flask import Flask, Response, render_template, request
from tqdm import tqdm

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import BeautifulSoup, extract_content, extract_date

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
        logger.info(f"[Preprocess] Index exists for {thread_dir}, skipping.")
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
        for el in soup.select("article.message"):
            content = extract_content(el).strip()
            if content:
                raw_posts.append(
                    {
                        "page": page,
                        "date": extract_date(el),
                        "content": clean_post_content(content),
                        "url": None,
                    }
                )
    logger.info(f"[Preprocess] Found {len(raw_posts)} posts to embed.")

    # 2) Load cache and prepare lists
    cache = load_cache()

    def post_hash(post):
        return hashlib.sha256(post["content"].encode("utf-8")).hexdigest()

    embeddings, posts, to_embed = [], [], []
    for post in raw_posts:
        h = post_hash(post)
        if h in cache and not force:
            embeddings.append(cache[h])
            posts.append(post)
        else:
            to_embed.append((h, post))

    logger.info(
        f"[Preprocess] {len(posts)} loaded from cache; {len(to_embed)} to embed."
    )

    # 3) Parallel embed the uncached posts
    with ThreadPoolExecutor(max_workers=8) as exe:
        futures = {exe.submit(embed_text, p["content"]): (h, p) for h, p in to_embed}
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
    logger.info(f"[Preprocess] Cache saved ({len(cache)} embeddings).")

    if not embeddings:
        logger.warning("[Preprocess] No embeddings → abort.")
        return

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
        logger.warning(f"[Search] Missing index/meta for {thread_dir}")
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
    logger.info(f"[Search] Returning {len(results)} posts for query.")
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
    thread_dir = get_thread_dir(thread_key)

    htmls = [f for f in os.listdir(thread_dir) if f.endswith(".html")]
    if url and (not htmls or refresh):
        last = detect_last_page(url)
        fetch_forum_pages(url, 1, last, save_dir=thread_dir)

    preprocess_thread(thread_dir, force=refresh)

    posts = find_relevant_posts(prompt, thread_dir)
    if not posts:
        context = "No relevant information found."
    else:
        context = "\n\n---\n\n".join(
            f"[Page {p['page']}] {p['date']}:\n{p['content']}" for p in posts
        )

    system = f"""You are an expert forum analyst. Use *only* the context. If no answer, say 'I do not know.'

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
            yield f"[Error: {e}]"
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
        shutil.rmtree(td)
        return f"Deleted {key}", 200
    return "Not found", 404


if __name__ == "__main__":
    logger.info("Starting Forum Wisdom Miner…")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)

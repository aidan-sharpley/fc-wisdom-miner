import html
import json
import logging
import os
import pickle
import re
import shutil
from typing import Dict, List

# Use hnswlib for fast ANN search
import hnswlib
import numpy as np
import requests
from flask import Flask, Response, render_template, request

# Add tqdm for progress bars
from tqdm import tqdm

# Your existing imports
from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import BeautifulSoup, extract_content, extract_date

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-a-secure-random-secret")
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_EMBED_API_URL = os.environ.get(
    "OLLAMA_EMBED_API_URL", "http://localhost:11434/api/embeddings"
)
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:1.5b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
BASE_TMP_DIR = os.environ.get("BASE_TMP_DIR", "tmp")
INDEX_META_NAME = "index_meta.pkl"
HNSW_INDEX_NAME = "index_hnsw.bin"

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


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    """
    1) Reads all HTML pages, cleans content, logs progress.
    2) Embeds with Ollama API, with progress bar.
    3) Builds HNSW index, saves metadata and index.
    """
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    hnsw_path = os.path.join(thread_dir, HNSW_INDEX_NAME)

    if not force and os.path.exists(meta_path) and os.path.exists(hnsw_path):
        logger.info(f"[Preprocess] Index already exists for {thread_dir}, skipping.")
        return

    logger.info(f"[Preprocess] Starting preprocessing for thread at: {thread_dir}")

    # 1) Load and clean posts
    raw_posts = []
    filenames = sorted(os.listdir(thread_dir))
    logger.info(f"[Preprocess] Found {len(filenames)} files in thread directory.")

    for filename in tqdm(filenames, desc="Parsing HTML pages"):
        logger.debug(f"[Preprocess] Inspecting file: {filename}")
        match = re.search(r"page[-_]?(\d+)\.html", filename)
        if not match:
            logger.debug(f"[Preprocess] Skipping non-page file: {filename}")
            continue
        page_num = int(match.group(1))

        with open(os.path.join(thread_dir, filename), encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        for post_el in soup.select("article.message"):
            content_str = extract_content(post_el)
            if content_str.strip():
                raw_posts.append(
                    {
                        "page": page_num,
                        "date": extract_date(post_el),
                        "content": clean_post_content(content_str),
                        "url": None,
                    }
                )
    logger.info(f"[Preprocess] Extracted {len(raw_posts)} raw posts.")

    # 2) Embed posts
    embeddings = []
    posts = []
    for post in tqdm(raw_posts, desc="Creating embeddings"):
        try:
            res = requests.post(
                OLLAMA_EMBED_API_URL,
                json={"model": OLLAMA_EMBED_MODEL, "prompt": post["content"]},
            )
            res.raise_for_status()
            emb = np.array(res.json()["embedding"], dtype="float32")
            embeddings.append(emb)
            posts.append(post)
        except Exception as e:
            logger.error(f"[Preprocess] Embedding failed for page {post['page']}: {e}")

    logger.info(
        f"[Preprocess] Successfully embedded {len(embeddings)}/{len(raw_posts)} posts."
    )

    if not embeddings:
        logger.warning("[Preprocess] No embeddings created; aborting index build.")
        return

    # 3) Build and save HNSW index
    dim = embeddings[0].shape[0]
    num_elements = len(embeddings)
    logger.info(
        f"[Preprocess] Building HNSW index (dim={dim}, elements={num_elements})."
    )
    p = hnswlib.Index(space="cosine", dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=200, M=32)
    p.add_items(np.vstack(embeddings))
    p.set_ef(50)

    logger.info(
        f"[Preprocess] Saving HNSW index to {hnsw_path} and metadata to {meta_path}."
    )
    p.save_index(hnsw_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"posts": posts, "dim": dim}, f)

    logger.info("[Preprocess] Done. Thread index ready.")


def find_relevant_posts(query: str, thread_dir: str, top_k: int = 5) -> List[Dict]:
    """
    Loads metadata & HNSW index, logs each step, embeds the query, and returns top_k posts.
    """
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    hnsw_path = os.path.join(thread_dir, HNSW_INDEX_NAME)

    if not os.path.exists(meta_path) or not os.path.exists(hnsw_path):
        logger.warning(f"[Search] Missing metadata or index for {thread_dir}.")
        return []

    logger.info(f"[Search] Loading metadata from {meta_path}.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    posts = meta["posts"]
    dim = meta["dim"]

    logger.info(f"[Search] Loading HNSW index (dim={dim}) from {hnsw_path}.")
    p = hnswlib.Index(space="cosine", dim=dim)
    p.load_index(hnsw_path)
    p.set_ef(50)

    logger.info("[Search] Embedding query.")
    try:
        res = requests.post(
            OLLAMA_EMBED_API_URL,
            json={"model": OLLAMA_EMBED_MODEL, "prompt": query},
        )
        res.raise_for_status()
        q_emb = np.array(res.json()["embedding"], dtype="float32")
    except Exception as e:
        logger.error(f"[Search] Query embedding failed: {e}")
        return []

    logger.info(f"[Search] Querying HNSW for top {top_k} results.")
    labels, distances = p.knn_query(q_emb, k=top_k)
    results = [posts[i] for i in labels[0] if i < len(posts)]
    logger.info(f"[Search] Retrieved {len(results)} relevant posts.")
    return results


@app.route("/")
def index():
    return render_template("index.html", threads=list_threads())


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    existing_thread = data.get("existing_thread", "").strip()
    url = data.get("url", "").strip()
    prompt = data.get("prompt", "").strip()
    refresh = data.get("refresh", False)

    if not prompt:
        return "Prompt is required", 400
    if not existing_thread and not url:
        return "Must provide a thread key or URL", 400

    thread_key = existing_thread or normalize_url(url).rstrip("/").split("/")[-1]
    thread_dir = get_thread_dir(thread_key)

    html_exists = any(f.endswith(".html") for f in os.listdir(thread_dir))
    if url and (not html_exists or refresh):
        last = detect_last_page(url)
        fetch_forum_pages(url, 1, last, save_dir=thread_dir)

    preprocess_thread(thread_dir, force=refresh)

    relevant = find_relevant_posts(prompt, thread_dir)
    if not relevant:
        context = "No relevant information found in the thread for this question."
    else:
        context = "\n\n---\n\n".join(
            f"[Page {p['page']}] {p['date']} (Link: {p.get('url', 'N/A')}):\n{p['content']}"
            for p in relevant
        )

    system_prompt = f"""
You are an expert forum analyst. Use *only* the provided context to answer. If the context does not contain the answer, respond with 'I do not know.'

---CONTEXT---
{context}
---END CONTEXT---

Question: {prompt}

Provide a concise answer rooted in the thread posts.
"""

    payload = {"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": True}

    def stream_response():
        try:
            with requests.post(OLLAMA_API_URL, json=payload, stream=True) as res:
                res.raise_for_status()
                for line in res.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"[Error: {e}]"
        yield "\n"

    return Response(stream_response(), content_type="text/plain")


@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    data = request.get_json(force=True)
    key = data.get("thread_key", "").strip()
    if not key or any(c in key for c in ["..", "/"]):
        return "Invalid thread key", 400
    td = os.path.join(BASE_TMP_DIR, key)
    if os.path.exists(td):
        shutil.rmtree(td)
        return f"Thread '{key}' deleted successfully.", 200
    return "Thread not found", 404


if __name__ == "__main__":
    logger.info("Starting app... Ensure your Ollama models are available.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)

import html
import json
import logging
import os
import pickle
import re
import shutil
from typing import Dict, List

import numpy as np
import requests
from flask import Flask, Response, render_template, request

# These imports are from your original files, ensure they are present
from fetch_forum import (
    detect_last_page,
    fetch_forum_pages,
)
from preprocess import (
    BeautifulSoup,
    extract_content,
    extract_date,
)

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-secret"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_API_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "deepseek-r1:1.5b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
BASE_TMP_DIR = "tmp"
INDEX_FILE_NAME = "index.pkl"

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


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    """
    Processes HTML files, cleans content, and creates a searchable vector index.
    """
    index_path = os.path.join(thread_dir, INDEX_FILE_NAME)
    if not force and os.path.exists(index_path):
        logger.info(f"Index already exists for {thread_dir}, skipping preprocessing.")
        return

    # Step 1: Extract posts from HTML
    raw_posts = []
    for filename in sorted(os.listdir(thread_dir)):
        logger.info(f"Processing file: {filename}")
        if filename.endswith(".html"):
            input_path = os.path.join(thread_dir, filename)
            match = re.search(r"page_(\d+)\.html", filename)
            page_num = int(match.group(1)) if match else -1

            with open(input_path, encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")
            post_elements = soup.select("article.message")

            for post_el in post_elements:
                content_str = extract_content(post_el)
                if content_str.strip():
                    raw_posts.append(
                        {
                            "page": page_num,
                            "date": extract_date(post_el),
                            "content": clean_post_content(content_str),
                        }
                    )

    all_posts = [p for p in raw_posts if p.get("content")]
    if not all_posts:
        logger.warning(f"No content found in thread {thread_dir}")
        return

    logger.info(f"Found {len(all_posts)} posts. Now creating embeddings...")

    # Step 2: Create embeddings for each post
    embeddings = []
    processed_posts = []  # Keep track of posts that successfully get embeddings

    for i, post in enumerate(all_posts):
        logger.info(f"Embedding post {i + 1}/{len(all_posts)}")
        try:
            res = requests.post(
                OLLAMA_EMBED_API_URL,
                json={"model": OLLAMA_EMBED_MODEL, "prompt": post["content"]},
            )
            res.raise_for_status()
            embedding = res.json()["embedding"]
            embeddings.append(embedding)
            processed_posts.append(post)  # Only add post if embedding was successful
        except requests.RequestException as e:
            logger.error(
                f"Failed to create embedding for post {i} (page {post.get('page', 'N/A')}): {e}. Skipping post."
            )
            # Do NOT append a zero array; simply skip this post.
            continue  # Move to the next post

    # Step 3: Save the posts and their embeddings to the index file
    if processed_posts:  # Only save if there are successfully embedded posts
        with open(index_path, "wb") as f:
            pickle.dump(
                {"posts": processed_posts, "embeddings": np.array(embeddings)}, f
            )
        logger.info(
            f"Successfully created and saved search index to {index_path} with {len(processed_posts)} posts."
        )
    else:
        logger.warning(
            f"No posts successfully embedded for {thread_dir}. Index not created."
        )


def find_relevant_posts(query: str, thread_index: Dict, top_k: int = 5) -> List[Dict]:
    """
    Finds the most relevant posts from a thread index based on a query.
    """
    if (
        not thread_index
        or "embeddings" not in thread_index
        or len(thread_index["embeddings"]) == 0
    ):
        return []

    # 1. Get embedding for the user's query
    try:
        res = requests.post(
            OLLAMA_EMBED_API_URL,
            json={"model": OLLAMA_EMBED_MODEL, "prompt": query},
        )
        res.raise_for_status()
        query_embedding = np.array(res.json()["embedding"])
    except requests.RequestException as e:
        logger.error(f"Failed to create embedding for query: {e}")
        return []

    # 2. Calculate cosine similarity
    post_embeddings = thread_index["embeddings"]
    dot_products = np.dot(post_embeddings, query_embedding)
    norms = np.linalg.norm(post_embeddings, axis=1) * np.linalg.norm(query_embedding)

    similarities = dot_products / np.where(norms == 0, 1e-9, norms)

    # 3. Get the indices of the top_k most similar posts
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # 4. Return the corresponding posts
    relevant_posts = [thread_index["posts"][i] for i in top_k_indices]
    return relevant_posts


@app.route("/")
def index():
    return render_template("index.html", threads=list_threads())


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    existing_thread = data.get("existing_thread", "").strip()
    url = data.get("url", "").strip()
    prompt = data.get("prompt", "").strip()
    refresh = data.get("refresh", False)

    if not prompt:
        return "Prompt is required", 400

    if existing_thread:
        thread_key = existing_thread
    elif url:
        url = normalize_url(url)
        thread_key = url.rstrip("/").split("/")[-1]
    else:
        return "Must provide a thread", 400

    if not thread_key:
        logger.error("URL or thread is missing or empty.")
        return "Invalid thread or URL", 400

    thread_dir = get_thread_dir(thread_key)
    index_path = os.path.join(thread_dir, INDEX_FILE_NAME)

    # Check if HTML files already exist in the thread directory
    html_files_exist = any(f.endswith(".html") for f in os.listdir(thread_dir))

    should_fetch_html = False
    if not html_files_exist:
        # Only fetch HTML if no HTML files are found (e.g., thread was just added or deleted)
        should_fetch_html = True
        logger.info(f"No HTML files found for {thread_key}. Fetching HTML pages...")
    elif refresh:
        # If HTML files exist AND refresh is true, we skip HTML fetching but force preprocessing
        logger.info(f"Refresh requested for {thread_key}. Skipping HTML re-fetch.")
    else:
        # If HTML files exist and no refresh, check if index exists to decide on preprocessing
        pass  # No explicit action here, `needs_preprocessing` logic below handles it.

    needs_preprocessing = False
    if should_fetch_html:
        # If we just fetched HTML, we definitely need to preprocess
        needs_preprocessing = True
        logger.debug(f"URL provided for fetching: {url}")
        last_page = detect_last_page(url)
        fetch_forum_pages(url, 1, last_page, save_dir=thread_dir)
    elif refresh:
        # If refresh is true AND HTML already exists, force reprocessing
        needs_preprocessing = True
    elif not os.path.exists(index_path):
        # If index doesn't exist but HTML does, preprocess to create index
        needs_preprocessing = True

    if needs_preprocessing:
        logger.info(f"Preprocessing thread and building index for {thread_key}...")
        preprocess_thread(thread_dir, force=True)  # Always force when reprocessing
    else:
        logger.info(
            f"Index already exists and no refresh requested for {thread_key}, skipping processing."
        )

    try:
        with open(index_path, "rb") as f:
            thread_index = pickle.load(f)
    except (IOError, pickle.PickleError) as e:
        logger.error(f"Could not load index file {index_path}: {e}")
        return (
            f"Error: Could not load data for thread '{thread_key}'. Please try refreshing.",
            500,
        )

    logger.info(f"Searching for context relevant to: '{prompt}'")
    relevant_posts = find_relevant_posts(prompt, thread_index)

    if not relevant_posts:
        context = "No relevant information found in the thread for this question."
    else:
        context = "\n\n---\n\n".join(
            f"[Page {p['page']}] {p['date']} (Link: {p.get('url', 'N/A')}):\n{p['content']}"
            for p in relevant_posts
        )

    logger.debug(f"Context being sent to LLM:\n{context}")

    system_prompt = f"""You are an expert forum analyst. Your task is to answer the user's question based *only* on the provided context. The context consists of the most relevant posts from a forum thread. Be concise and helpful.

---CONTEXT---
{context}
---END CONTEXT---

Based on the context above, answer the following question:
{prompt}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt,
        "stream": True,
    }

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
                            logger.warning(f"Skipping malformed line: {line}")
        except requests.RequestException as e:
            logger.error(f"Stream exception: {e}")
            yield f"[Streaming error: {e}]"
        yield "\n"

    return Response(stream_response(), content_type="text/plain")


@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    data = request.get_json()
    thread_key = data.get("thread_key", "").strip()
    if not thread_key:
        return "Thread key is required", 400
    if ".." in thread_key or "/" in thread_key:
        return "Invalid thread key", 400
    thread_dir = os.path.join(BASE_TMP_DIR, thread_key)
    if os.path.exists(thread_dir):
        try:
            shutil.rmtree(thread_dir)
            logger.info(f"Deleted thread directory: {thread_dir}")
            return f"Thread '{thread_key}' deleted successfully.", 200
        except Exception as e:
            logger.error(f"Error deleting thread {thread_key}: {e}")
            return f"Error deleting thread: {e}", 500
    else:
        return "Thread not found", 404


if __name__ == "__main__":
    logger.info(
        f"Make sure you have the Ollama embedding model: 'ollama pull {OLLAMA_EMBED_MODEL}'"
    )
    logger.info("Launching app at http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)

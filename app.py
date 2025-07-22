import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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

# Timeouts and sizes
API_TIMEOUT = 10  # seconds for HTTP calls
BATCH_RERANK_TIMEOUT = 30
QUERY_RERANK_SIZE = 25
FINAL_TOP_K = 7

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-secure-secret")


# -------------------- Utility Functions --------------------
def post_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else "https://" + url


def get_thread_dir(thread_key: str) -> str:
    path = os.path.join(BASE_TMP_DIR, thread_key)
    os.makedirs(path, exist_ok=True)
    return path


def list_threads() -> List[str]:
    if not os.path.isdir(BASE_TMP_DIR):
        return []
    return sorted(
        [
            d
            for d in os.listdir(BASE_TMP_DIR)
            if os.path.isdir(os.path.join(BASE_TMP_DIR, d))
        ]
    )


def clean_post_content(raw: str) -> str:
    text = BeautifulSoup(raw, "html.parser").get_text()
    text = re.sub(r"(?:^|\n)\s*\w+ said:\s*(Click to expand\.\.\.)?", "", text)
    text = re.sub(r"Click to expand\.\.\.", "", text)
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


# -------------------- HTML Parsing --------------------
def extract_date(el) -> str:
    tag = el.find("time")
    return tag["datetime"] if tag and tag.has_attr("datetime") else "unknown-date"


def extract_author(el) -> str:
    return el.get("data-author", "unknown-author")


def extract_content(el) -> str:
    for selector in ["div.message-userContent .bbWrapper", ".message-body .bbWrapper"]:
        node = el.select_one(selector)
        if node:
            return node.get_text(separator="\n").strip()
    return ""


def extract_post_url(el, base_url: str) -> str:
    link = el.select_one("ul.message-attribution-opposite a[href]")
    if link and link.has_attr("href"):
        return requests.compat.urljoin(base_url, link["href"])
    data = el.get("data-content", "")
    if data.startswith("post-"):
        pid = data.split("-")[1]
        return f"{base_url}posts/{pid}/"
    return f"{base_url}#unknown-post"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    link = soup.find("link", rel="canonical")
    if link and link.has_attr("href"):
        return re.sub(r"page-\d+/?$", "", link["href"]).rstrip("/") + "/"
    return ""


# -------------------- Web Scraping --------------------
def detect_last_page(url: str) -> int:
    try:
        r = requests.get(url, timeout=API_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        inp = soup.select_one("input.js-pageJumpPage[max]")
        if inp and inp.has_attr("max"):
            return int(inp["max"])
        last = soup.select_one(".pageNav-main li:last-child a")
        if last and last.text.isdigit():
            return int(last.text)
    except Exception as e:
        logger.warning(f"Could not detect last page for {url}: {e}")
    return 1


def fetch_forum_pages(base_url: str, last_page: int, save_dir: str):
    base = re.sub(r"page-\d+/?$", "", base_url).rstrip("/")
    for i in range(1, last_page + 1):
        out = os.path.join(save_dir, f"page{i}.html")
        if os.path.exists(out):
            continue
        try:
            r = requests.get(f"{base}/page-{i}", timeout=API_TIMEOUT)
            r.raise_for_status()
            with open(out, "w", encoding="utf-8") as f:
                f.write(r.text)
            time.sleep(0.5)
            logger.info(f"Saved page {i}/{last_page}")
        except Exception as e:
            logger.error(f"Error fetching page {i}: {e}")
            break


# -------------------- Embeddings & Cache --------------------
def load_cache() -> Dict[str, np.ndarray]:
    if os.path.exists(CACHE_PATH):
        try:
            return pickle.load(open(CACHE_PATH, "rb"))
        except Exception:
            logger.warning("Embeddings cache corrupted; resetting.")
    return {}


def save_cache(cache: Dict[str, np.ndarray]):
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    pickle.dump(cache, open(CACHE_PATH, "wb"))


def embed_text(text: str) -> np.ndarray:
    r = requests.post(
        OLLAMA_EMBED_API_URL,
        json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=API_TIMEOUT,
    )
    r.raise_for_status()
    return np.array(r.json().get("embedding", []), dtype="float32")


# -------------------- Preprocessing --------------------
def preprocess_thread(thread_dir: str, force: bool = False):
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    index_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    meta_index = os.path.join(thread_dir, METADATA_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")

    if not force and all(
        os.path.exists(p) for p in [meta_path, index_path, meta_index, posts_dir]
    ):
        logger.info(f"Preprocessing exists for {thread_dir}, skipping.")
        return

    if force and os.path.exists(posts_dir):
        shutil.rmtree(posts_dir)
    os.makedirs(posts_dir, exist_ok=True)

    htmls = sorted(
        [f for f in os.listdir(thread_dir) if f.endswith(".html")],
        key=lambda x: int(re.search(r"page(\d+)", x).group(1)),
    )
    raw_posts, metadata = [], []
    for fn in htmls:
        page = int(re.search(r"page(\d+)", fn).group(1))
        soup = BeautifulSoup(
            open(os.path.join(thread_dir, fn), encoding="utf-8"), "html.parser"
        )
        base = extract_canonical_url(soup)
        for el in soup.select("article.message"):
            content = clean_post_content(extract_content(el))
            if not content:
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
                {"page": page, "date": post["date"], "author": post["author"]}
            )

    with open(meta_index, "w") as f:
        json.dump(metadata, f, indent=2)

    cache = load_cache()
    to_embed = [p for p in raw_posts if force or post_hash(p["content"]) not in cache]
    logger.info(
        f"Embedding {len(to_embed)} new posts (cached: {len(raw_posts) - len(to_embed)})"
    )
    if to_embed:
        with ThreadPoolExecutor() as ex:
            futures = {ex.submit(embed_text, p["content"]): p for p in to_embed}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Embedding posts"
            ):
                try:
                    emb = future.result()
                    cache[post_hash(futures[future]["content"])] = emb
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
        save_cache(cache)

    vectors = []
    for idx, post in enumerate(raw_posts):
        h = post_hash(post["content"])
        vectors.append(cache[h])
        with open(os.path.join(posts_dir, f"{idx}.json"), "w") as f:
            json.dump(post, f)

    dim = vectors[0].shape[0]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=32)
    index.add_items(np.vstack(vectors))
    index.save_index(index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"dim": dim, "count": len(vectors)}, f)
    logger.info(f"Built HNSW index with {len(vectors)} items.")


# -------------------- HyDE & Batched Rerank --------------------
def generate_hyde(query: str) -> str:
    prompt = f"Write a concise answer to help retrieve relevant forum posts.\n\nQuestion: {query}"
    try:
        r = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            timeout=API_TIMEOUT,
        )
        r.raise_for_status()
        # Attempt JSON parse, fallback to raw text
        try:
            return r.json().get("response", query)
        except ValueError:
            return r.text.strip() or query
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query


def batch_rerank(query: str, posts: List[Dict]) -> List[Tuple[int, Dict]]:
    lines = [
        f"Question: {query}",
        "Rate relevance of each post from 0-10, respond ONLY with numbered list:",
    ]
    for i, p in enumerate(posts, 1):
        snippet = p["content"][:300].replace("\n", " ")
        lines.append(f"{i}. {snippet}...")
    prompt = "\n".join(lines)
    try:
        r = requests.post(
            OLLAMA_API_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt},
            timeout=BATCH_RERANK_TIMEOUT,
        )
        r.raise_for_status()
        # Fallback JSON/text handling
        text = ""
        try:
            text = r.json().get("response", "")
        except ValueError:
            text = r.text
        scores = [int(m.group(1)) for m in re.finditer(r"\d+\.\s*(\d+)", text)]
        if len(scores) != len(posts):
            raise ValueError("Mismatch between scores and posts count")
        return list(zip(scores, posts))
    except Exception as e:
        logger.warning(f"Rerank batch failed: {e}")
        return [(0, p) for p in posts]


# -------------------- Search Logic --------------------
def find_posts_by_metadata(
    thread_dir: str,
    author: Optional[str] = None,
    page: Optional[int] = None,
    date: Optional[str] = None,
) -> List[Dict]:
    meta_path = os.path.join(thread_dir, METADATA_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")
    if not os.path.exists(meta_path):
        return []
    metadata = json.load(open(meta_path))
    matches = []
    for idx, m in enumerate(metadata):
        if author and author.lower() in m.get("author", "").lower():
            matches.append(idx)
        elif page and page == m.get("page"):
            matches.append(idx)
        elif date and date.lower() in m.get("date", "").lower():
            matches.append(idx)
    results = []
    for i in matches[:15]:
        try:
            results.append(json.load(open(os.path.join(posts_dir, f"{i}.json"))))
        except:
            continue
    logger.info(f"Metadata search found {len(results)} posts.")
    return results


def find_relevant_posts(
    query: str, thread_dir: str, top_k: int = FINAL_TOP_K
) -> List[Dict]:
    meta = pickle.load(open(os.path.join(thread_dir, INDEX_META_NAME), "rb"))
    idx = hnswlib.Index(space="cosine", dim=meta["dim"])
    idx.load_index(os.path.join(thread_dir, HNSW_INDEX_NAME))

    hyde_q = generate_hyde(query)
    q_emb = embed_text(hyde_q)
    labels, _ = idx.knn_query(q_emb, k=QUERY_RERANK_SIZE)

    posts_dir = os.path.join(thread_dir, "posts")
    candidates = []
    for i in labels[0]:
        path = os.path.join(posts_dir, f"{i}.json")
        if os.path.exists(path):
            candidates.append(json.load(open(path)))

    scored = batch_rerank(query, candidates)
    scored.sort(key=lambda x: x[0], reverse=True)
    final = [p for _, p in scored[:top_k]]
    logger.info(f"Selected {len(final)} posts after rerank.")
    return final


# -------------------- Flask Routes --------------------
@app.route("/")
def index():
    return render_template("index.html", threads=list_threads())


@app.route("/preprocess", methods=["POST"])
def preprocess():
    data = request.json or {}
    thread = data.get("thread_key", "").strip()
    url = data.get("url", "").strip()
    refresh = data.get("refresh", False)
    if not thread and not url:
        return jsonify({"error": "thread_key or url required"}), 400
    thread_key = thread or normalize_url(url).rstrip("/").split("/")[-1]
    dir_ = get_thread_dir(thread_key)
    if url and not any(f.endswith(".html") for f in os.listdir(dir_)):
        last = detect_last_page(url)
        fetch_forum_pages(url, last, dir_)
    preprocess_thread(dir_, force=refresh)
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True) or {}
    prompt = data.get("prompt", "").strip()
    url = data.get("url", "").strip()
    existing = data.get("existing_thread", "").strip()
    refresh = data.get("refresh", False)
    if not prompt or not (url or existing):
        return "Prompt and URL/existing thread required.", 400

    thread_key = existing or normalize_url(url).rstrip("/").split("/")[-1]
    thread_dir = get_thread_dir(thread_key)

    # Metadata-first shortcuts
    lower = prompt.lower()
    posts: List[Dict] = []
    if "first post" in lower:
        try:
            posts = [json.load(open(os.path.join(thread_dir, "posts", "0.json")))]
        except:
            posts = []
    elif "last post" in lower:
        try:
            count = pickle.load(open(os.path.join(thread_dir, INDEX_META_NAME), "rb"))[
                "count"
            ]
            posts = [
                json.load(open(os.path.join(thread_dir, "posts", f"{count - 1}.json")))
            ]
        except:
            posts = []
    else:
        author_m = re.search(r"posts by\s+([\w-]+)", lower)
        page_m = re.search(r"on page\s+(\d+)", lower)
        date_m = re.search(r"from\s+([A-Za-z]+ \d{1,2},? \d{4})", lower)
        if author_m:
            posts = find_posts_by_metadata(thread_dir, author=author_m.group(1))
        elif page_m:
            posts = find_posts_by_metadata(thread_dir, page=int(page_m.group(1)))
        elif date_m:
            posts = find_posts_by_metadata(thread_dir, date=date_m.group(1))

    # Fallback to semantic search
    if not posts:
        posts = find_relevant_posts(prompt, thread_dir)

    context = (
        "\n\n".join(
            f"--- Post by {p['author']} on {p['date']} (Page {p['page']}) ---\nURL: {p['url']}\n\n{p['content']}"
            for p in posts
        )
        or "No relevant information found."
    )

    system = (
        "You are an expert forum analyst. Use *only* the provided context below to answer the question.\n\n"
        f"---CONTEXT---\n{context}\n---END CONTEXT---\n\nQuestion: {prompt}"
    )

    def stream_response():
        try:
            payload = {"model": OLLAMA_MODEL, "prompt": system, "stream": True}
            with requests.post(
                OLLAMA_API_URL, json=payload, stream=True, timeout=60
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        chunk = json.loads(line)
                        yield chunk.get("response", "")
        except Exception as e:
            logger.error(f"Error streaming from LLM: {e}")
            yield "[Error communicating with the language model]\n"

    return Response(stream_response(), mimetype="text/plain")


@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    data = request.json or {}
    key = data.get("thread_key", "").strip()
    if not key or any(c in key for c in ("..", "/")):
        return "Invalid thread key", 400
    td = os.path.join(BASE_TMP_DIR, key)
    if os.path.exists(td):
        shutil.rmtree(td)
        return f"Deleted {key}", 200
    return "Not found", 404


if __name__ == "__main__":
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=False)

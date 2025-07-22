import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

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

API_TIMEOUT = 10  # HTTP timeout
MAX_RETRIES = 3  # HTTP retries
RETRY_BACKOFF = 2  # Exponential backoff
CHUNK_SIZE = 1000  # chars per embed chunk
CHUNK_OVERLAP = 200  # chunk overlap
QUERY_RERANK_SIZE = 25
BATCH_RERANK_TIMEOUT = 30
FINAL_TOP_K = 7

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "replace-with-secure-secret")


# -------------------- Utilities --------------------
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
        d
        for d in os.listdir(BASE_TMP_DIR)
        if os.path.isdir(os.path.join(BASE_TMP_DIR, d))
    )


def clean_post_content(raw: str) -> str:
    text = BeautifulSoup(raw, "html.parser").get_text()
    text = re.sub(r"(?:^|\n)\s*\w+ said:\s*(Click to expand\.{3})?", "", text)
    text = re.sub(r"Click to expand\.{3}", "", text)
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
    for sel in ["div.message-userContent .bbWrapper", ".message-body .bbWrapper"]:
        node = el.select_one(sel)
        if node:
            return node.get_text(separator="\n").strip()
    return ""


def extract_post_url(el, base: str) -> str:
    link = el.select_one("ul.message-attribution-opposite a[href]")
    if link and link.has_attr("href"):
        return requests.compat.urljoin(base, link["href"])
    data = el.get("data-content", "")
    if data.startswith("post-"):
        pid = data.split("-")[1]
        return f"{base}posts/{pid}/"
    return f"{base}#unknown-post"


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
        fn = os.path.join(save_dir, f"page{i}.html")
        if os.path.exists(fn):
            continue
        try:
            r = requests.get(f"{base}/page-{i}", timeout=API_TIMEOUT)
            r.raise_for_status()
            with open(fn, "w", encoding="utf-8") as f:
                f.write(r.text)
            time.sleep(0.5)
            logger.info(f"Saved page {i}/{last_page}")
        except Exception as e:
            logger.error(f"Error fetching page {i}: {e}")
            break


# -------------------- HTTP with Retries --------------------
def http_post(url: str, json_data: dict, timeout: int):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, json=json_data, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Request to {url} failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** (attempt - 1))
            else:
                raise


# -------------------- Embedding --------------------
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


def chunk_text(text: str) -> List[str]:
    chunks, start, length = [], 0, len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_text(text: str) -> np.ndarray:
    cache = load_cache()
    h = post_hash(text)
    if h in cache:
        return cache[h]
    chunks = chunk_text(text) or [text]
    vectors = []
    for c in chunks:
        r = http_post(
            OLLAMA_EMBED_API_URL,
            {"model": OLLAMA_EMBED_MODEL, "prompt": c},
            API_TIMEOUT,
        )
        emb = np.array(r.json().get("embedding", []), dtype="float32")
        vectors.append(emb)
    vec = np.mean(vectors, axis=0)
    cache[h] = vec
    save_cache(cache)
    return vec


# -------------------- Preprocessing --------------------
def preprocess_thread(thread_dir: str, force: bool = False):
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    idx_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    meta_index = os.path.join(thread_dir, METADATA_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")

    if not force and all(
        os.path.exists(p) for p in [meta_path, idx_path, meta_index, posts_dir]
    ):
        logger.info(f"Preprocessing exists for {thread_dir}, skipping.")
        return

    if force and os.path.exists(posts_dir):
        shutil.rmtree(posts_dir)
    os.makedirs(posts_dir, exist_ok=True)

    htmls = sorted(f for f in os.listdir(thread_dir) if f.endswith(".html"))
    raw_posts, metadata = [], []
    for fn in htmls:
        page = int(re.search(r"page(\d+)", fn).group(1))
        with open(os.path.join(thread_dir, fn), encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
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
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Embedding posts"
            ):
                try:
                    emb = fut.result()
                    cache[post_hash(futures[fut]["content"])] = emb
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
        save_cache(cache)

    vectors = []
    for i, post in enumerate(raw_posts):
        vec = cache[post_hash(post["content"])]
        vectors.append(vec)
        with open(os.path.join(posts_dir, f"{i}.json"), "w") as f:
            json.dump(post, f)

    dim = vectors[0].shape[0]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=32)
    index.add_items(np.vstack(vectors))
    index.save_index(idx_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"dim": dim, "count": len(vectors)}, f)
    logger.info(f"Built HNSW index with {len(vectors)} items.")


# -------------------- HyDE --------------------
def generate_hyde(query: str) -> str:
    prompt = f"Write a concise answer to help retrieve relevant forum posts.\n\nQuestion: {query}"
    try:
        r = http_post(
            OLLAMA_API_URL, {"model": OLLAMA_MODEL, "prompt": prompt}, API_TIMEOUT
        )
        text = r.text
        try:
            data = json.loads(text)
            return data.get("response", query)
        except Exception:
            pass
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                return obj.get("response", query)
            except Exception:
                continue
        return text.strip() or query
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return query


# -------------------- Rerank --------------------
def batch_rerank(query: str, posts: List[Dict]) -> List[Tuple[int, Dict]]:
    lines = [f"Question: {query}", "Rate relevance 0-10, numbered list:"]
    for i, p in enumerate(posts, 1):
        lines.append(f"{i}. {p['content'][:300].replace('\n', ' ')}...")
    prompt = "\n".join(lines)
    try:
        r = http_post(
            OLLAMA_API_URL,
            {"model": OLLAMA_MODEL, "prompt": prompt},
            BATCH_RERANK_TIMEOUT,
        )
        data = json.loads(r.text) if r.text.strip().startswith("{") else None
        text = data.get("response", "") if data else r.text
        scores = [int(m.group(1)) for m in re.finditer(r"\d+\.\s*(\d+)", text)]
        if len(scores) != len(posts):
            raise ValueError("Score count mismatch")
        return list(zip(scores, posts))
    except Exception as e:
        logger.warning(f"Rerank failed: {e}")
        return [(0, p) for p in posts]


# -------------------- Search --------------------
def find_posts_by_metadata(
    thread_dir: str, author=None, page=None, date=None
) -> List[Dict]:
    path = os.path.join(thread_dir, METADATA_INDEX_NAME)
    posts_dir = os.path.join(thread_dir, "posts")
    if not os.path.exists(path):
        return []
    meta = json.load(open(path))
    idxs = []
    for i, m in enumerate(meta):
        if author and author.lower() in m.get("author", "").lower():
            idxs.append(i)
        elif page and page == m.get("page"):
            idxs.append(i)
        elif date and date.lower() in m.get("date", "").lower():
            idxs.append(i)
    res = []
    for i in idxs[:15]:
        try:
            res.append(json.load(open(os.path.join(posts_dir, f"{i}.json"))))
        except:
            pass
    return res


def find_relevant_posts(query: str, thread_dir: str, top_k=FINAL_TOP_K) -> List[Dict]:
    meta_path = os.path.join(thread_dir, INDEX_META_NAME)
    idx_path = os.path.join(thread_dir, HNSW_INDEX_NAME)
    if not os.path.exists(meta_path) or not os.path.exists(idx_path):
        return []
    meta = pickle.load(open(meta_path, "rb"))
    count, dim = meta.get("count", 0), meta.get("dim", 0)
    if count == 0 or dim == 0:
        return []
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.load_index(idx_path)
    hyde_q = generate_hyde(query)
    q_emb = embed_text(hyde_q)
    if q_emb.size != dim:
        logger.warning(f"Embedding dim {q_emb.size} != {dim}")
        return []
    labels, _ = idx.knn_query(q_emb, k=min(QUERY_RERANK_SIZE, count))
    posts_dir = os.path.join(thread_dir, "posts")
    cands = [
        json.load(open(os.path.join(posts_dir, f"{i}.json")))
        for i in labels[0]
        if os.path.exists(os.path.join(posts_dir, f"{i}.json"))
    ]
    scored = batch_rerank(query, cands)
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


# -------------------- Routes --------------------
@app.route("/")
def index_route():
    return render_template("index.html", threads=list_threads())


@app.route("/preprocess", methods=["POST"])
def preprocess_route():
    data = request.json or {}
    thread, url, refresh = (
        data.get("thread_key", ""),
        data.get("url", ""),
        data.get("refresh", False),
    )
    if not thread and not url:
        return jsonify({"error": "thread_key or url required"}), 400
    key = thread or normalize_url(url).rstrip("/").split("/")[-1]
    d = get_thread_dir(key)
    htmls = [f for f in os.listdir(d) if f.endswith(".html")]
    if url and (refresh or not htmls):
        lp = detect_last_page(url)
        fetch_forum_pages(url, lp, d)
    preprocess_thread(d, force=refresh)
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask_route():
    data = request.get_json(force=True) or {}
    prompt, url, existing, refresh = (
        data.get("prompt", ""),
        data.get("url", ""),
        data.get("existing_thread", ""),
        data.get("refresh", False),
    )
    if not prompt or not (url or existing):
        return "Prompt and URL/existing required", 400
    key = existing or normalize_url(url).rstrip("/").split("/")[-1]
    d = get_thread_dir(key)
    htmls = [f for f in os.listdir(d) if f.endswith(".html")]
    if url and (refresh or not htmls):
        lp = detect_last_page(url)
        fetch_forum_pages(url, lp, d)
    preprocess_thread(d, force=refresh)
    lower = prompt.lower()
    posts = []
    if "first post" in lower:
        try:
            posts = [json.load(open(os.path.join(d, "posts", "0.json")))]
        except:
            posts = []
    elif "last post" in lower:
        try:
            cnt = pickle.load(open(os.path.join(d, INDEX_META_NAME), "rb"))["count"]
            if cnt > 0:
                posts = [json.load(open(os.path.join(d, "posts", f"{cnt - 1}.json")))]
        except:
            posts = []
    else:
        am, pm, dm = (
            re.search(r"posts by\s+([\w-]+)", lower),
            re.search(r"on page\s+(\d+)", lower),
            re.search(r"from\s+(.+)", lower),
        )
        if am:
            posts = find_posts_by_metadata(d, author=am.group(1))
        elif pm:
            posts = find_posts_by_metadata(d, page=int(pm.group(1)))
        elif dm:
            posts = find_posts_by_metadata(d, date=dm.group(1))
    if not posts:
        posts = find_relevant_posts(prompt, d)
    context = (
        "\n\n".join(
            f"--- Post by {p['author']} on {p['date']} (Page {p['page']}) ---\nURL: {p['url']}\n\n{p['content']}"
            for p in posts
        )
        or "No relevant information found."
    )
    system = f"You are an expert forum analyst. Use *only* the provided context below to answer the question.\n\n---CONTEXT---\n{context}\n---END CONTEXT---\n\nQuestion: {prompt}"

    def stream():
        try:
            r = requests.post(
                OLLAMA_API_URL,
                json={"model": OLLAMA_MODEL, "prompt": system, "stream": True},
                stream=True,
                timeout=60,
            )
            r.raise_for_status()
            for L in r.iter_lines(decode_unicode=True):
                if L:
                    yield json.loads(L).get("response", "")
        except Exception as e:
            logger.error(f"LLM stream error: {e}")
            yield "[Error communicating with the language model]\n"

    return Response(stream(), mimetype="text/plain")


@app.route("/delete_thread", methods=["POST"])
def delete_route():
    data = request.json or {}
    key = data.get("thread_key", "").strip()
    if not key or any(c in key for c in ("..", "/")):
        return "Invalid key", 400
    td = os.path.join(BASE_TMP_DIR, key)
    if os.path.exists(td):
        shutil.rmtree(td)
        return f"Deleted {key}", 200
    return "Not found", 404


if __name__ == "__main__":
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=False)

import html
import json
import logging
import os
import re
import shutil  # <-- Import shutil for directory deletion
from typing import List

import requests
from flask import Flask, Response, render_template, request

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import clean_html_file

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-secret"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"
BASE_TMP_DIR = "tmp"

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s - %(message)s"
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
    combined_path = os.path.join(thread_dir, "combined.jsonl")
    all_posts = []

    for filename in sorted(os.listdir(thread_dir)):
        if filename.endswith(".html"):
            input_path = os.path.join(thread_dir, filename)
            output_path = os.path.join(thread_dir, filename.replace(".html", ".txt"))
            if force or not os.path.exists(output_path):
                logger.debug(f"Cleaning {input_path}")
                match = re.search(r"page_(\d+)\.html", filename)
                page_number = int(match.group(1)) if match else -1
                clean_html_file(input_path, output_path, page_number)

    for filename in sorted(os.listdir(thread_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(thread_dir, filename), encoding="utf-8") as f:
                for line in f:
                    try:
                        post = json.loads(line.strip())
                        content = post.get("content", "")
                        if not content:
                            continue
                        cleaned = clean_post_content(content)
                        if not cleaned:
                            continue
                        post["content"] = cleaned
                        all_posts.append(post)
                    except Exception as e:
                        logger.warning(f"Skipping invalid post in {filename}: {e}")

    with open(combined_path, "w", encoding="utf-8") as f:
        for post in all_posts:
            compact = {
                "date": post["date"],
                "page": post["page"],
                "content": post["content"],
            }
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    logger.info(f"Optimally combined {len(all_posts)} posts into {combined_path}")


def load_thread_text(thread_dir: str) -> str:
    combined_path = os.path.join(thread_dir, "combined.jsonl")
    lines = []

    with open(combined_path, encoding="utf-8") as f:
        for line in f:
            try:
                post = json.loads(line.strip())
                lines.append(
                    f"[Page {post['page']}] {post['date']}:\n{post['content']}"
                )
            except Exception as e:
                logger.warning(f"Malformed line: {e}")
    return "\n\n".join(lines)


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
        thread_dir = get_thread_dir(thread_key)
        if refresh:
            preprocess_thread(thread_dir, force=True)
    elif url:
        url = normalize_url(url)
        thread_key = url.rstrip("/").split("/")[-1]
        thread_dir = get_thread_dir(thread_key)
        if refresh or not any(f.endswith(".html") for f in os.listdir(thread_dir)):
            last_page = detect_last_page(url)
            fetch_forum_pages(url, 1, last_page, save_dir=thread_dir)
            preprocess_thread(thread_dir, force=True)
    else:
        return "Must provide a thread", 400

    context = load_thread_text(thread_dir)
    logger.debug(f"Loaded context length: {len(context)}")

    # -- REFINED SYSTEM PROMPT FOR BETTER ACCURACY AND SPEED --
    system_prompt = f"""You are an expert forum analyst. Your task is to answer the user's question based *only* on the provided context from a forum thread. Be concise and helpful.

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
                for line in res.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                text = chunk["response"]
                                yield text
                        except Exception as e:
                            logger.warning(f"Chunk error: {e} -- line: {line}")
        except Exception as e:
            logger.error(f"Stream exception: {e}")
            yield f"[Streaming error: {e}]"
        yield "\n"

    return Response(stream_response(), content_type="text/plain")


# -- NEW ENDPOINT TO DELETE A THREAD --
@app.route("/delete_thread", methods=["POST"])
def delete_thread():
    data = request.get_json()
    thread_key = data.get("thread_key", "").strip()
    if not thread_key:
        return "Thread key is required", 400

    # Basic security check to prevent directory traversal
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
    logger.info("Launching app at http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)

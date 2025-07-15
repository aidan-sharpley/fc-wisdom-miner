import json
import logging
import os
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


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    for filename in os.listdir(thread_dir):
        if filename.endswith(".html"):
            input_path = os.path.join(thread_dir, filename)
            output_path = os.path.join(thread_dir, filename.replace(".html", ".txt"))
            if force or not os.path.exists(output_path):
                logger.debug(f"Cleaning {input_path}")
                clean_html_file(input_path, output_path)


def load_thread_text(thread_dir: str) -> str:
    texts = []
    for fname in sorted(os.listdir(thread_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(thread_dir, fname), encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)


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
    elif url:
        url = normalize_url(url)
        thread_key = url.rstrip("/").split("/")[-1]
        thread_dir = get_thread_dir(thread_key)
        if refresh or not any(f.endswith(".html") for f in os.listdir(thread_dir)):
            last_page = detect_last_page(url)
            fetch_forum_pages(url, 1, last_page, save_dir=thread_dir)
            preprocess_thread(thread_dir, force=True)
        else:
            preprocess_thread(thread_dir, force=False)
    else:
        return "Must provide a thread", 400

    context = load_thread_text(thread_dir)
    logger.debug(f"Loaded context length: {len(context)}")

    system_prompt = (
        "You are an expert forum analyst. Based on the following anonymized forum content, "
        "answer the user's question with clear reasoning.\n\n"
        f"{context}\n\n"
        f"Question: {prompt}"
    )

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


if __name__ == "__main__":
    logger.info("Launching app at http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)

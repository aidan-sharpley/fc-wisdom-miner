import logging
import os
from typing import List

import requests
from flask import (
    Flask,
    Response,
    render_template,
    request,
)

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import clean_html_file

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-secret"  # For flashing messages

OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"
BASE_TMP_DIR = "tmp"

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        logger.debug(f"URL missing scheme, adding https:// prefix: {url}")
        return "https://" + url
    return url


def get_thread_dir(thread_key: str) -> str:
    thread_dir = os.path.join(BASE_TMP_DIR, thread_key)
    os.makedirs(thread_dir, exist_ok=True)
    logger.debug(f"Thread directory resolved: {thread_dir}")
    return thread_dir


def list_threads() -> List[str]:
    if not os.path.exists(BASE_TMP_DIR):
        logger.debug(f"Base tmp dir '{BASE_TMP_DIR}' does not exist.")
        return []
    dirs = [
        d
        for d in os.listdir(BASE_TMP_DIR)
        if os.path.isdir(os.path.join(BASE_TMP_DIR, d))
    ]
    logger.debug(f"Available threads: {dirs}")
    return sorted(dirs)


def preprocess_thread(thread_dir: str, force: bool = False) -> None:
    logger.info(f"Preprocessing thread in {thread_dir}, force={force}")
    for filename in os.listdir(thread_dir):
        if filename.endswith(".html"):
            input_path = os.path.join(thread_dir, filename)
            output_path = os.path.join(thread_dir, filename.replace(".html", ".txt"))
            if force or not os.path.exists(output_path):
                logger.debug(f"Cleaning HTML file {input_path} -> {output_path}")
                clean_html_file(input_path, output_path)
            else:
                logger.debug(f"Skipping preprocessing for existing {output_path}")


def load_thread_text(thread_dir: str) -> str:
    texts: List[str] = []
    logger.debug(f"Loading .txt files from {thread_dir}")
    for fname in sorted(os.listdir(thread_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(thread_dir, fname)
            logger.debug(f"Reading {path}")
            with open(path, encoding="utf-8") as f:
                texts.append(f.read())
    combined_text = "\n\n".join(texts)
    logger.debug(f"Loaded combined text length: {len(combined_text)}")
    return combined_text


@app.route("/", methods=["GET"])
def index():
    threads = list_threads()
    return render_template("index.html", threads=threads)


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
        logger.info(f"Using existing thread: {thread_key}")
    elif url:
        url = normalize_url(url)
        thread_key = url.rstrip("/").split("/")[-1]
        thread_dir = get_thread_dir(thread_key)
        logger.info(f"Using URL thread: {url}")

        # Fetch and preprocess if refresh requested or no HTML files found
        if refresh or not any(f.endswith(".html") for f in os.listdir(thread_dir)):
            logger.info("Fetching forum pages due to refresh or no HTML present")
            last_page = detect_last_page(url)
            logger.info(f"Detected last page: {last_page}")
            fetch_forum_pages(url, start=1, end=last_page, save_dir=thread_dir)
            preprocess_thread(thread_dir, force=True)
        else:
            preprocess_thread(thread_dir, force=False)
    else:
        return "Must provide either existing_thread or url", 400

    context = load_thread_text(thread_dir)

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

    try:
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as res:
            if not res.ok:
                logger.error(f"Ollama API error: {res.status_code} {res.text}")
                return "[Error communicating with Ollama]", 500

            def generate():
                import json

                buffer = ""
                for chunk in res.iter_content(chunk_size=1024):
                    text = chunk.decode("utf-8")
                    buffer += text
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip():
                            logger.debug(f"Received line: {line}")
                            try:
                                chunk_json = json.loads(line)
                                logger.debug(f"Parsed chunk JSON: {chunk_json}")
                                if "response" in chunk_json:
                                    yield chunk_json["response"]
                            except Exception as e:
                                logger.warning(
                                    f"Could not parse line: {line} error: {e}"
                                )
                # Try parse any remaining buffer
                if buffer.strip():
                    try:
                        chunk_json = json.loads(buffer)
                        if "response" in chunk_json:
                            yield chunk_json["response"]
                    except Exception as e:
                        logger.warning(
                            f"Could not parse final buffer: {buffer} error: {e}"
                        )
                yield "\n"

            return Response(generate(), content_type="text/plain")

    except Exception as e:
        logger.exception("Exception during Ollama request")
        return f"[Error processing request: {e}]", 500


if __name__ == "__main__":
    logger.info("Starting Flask app on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)

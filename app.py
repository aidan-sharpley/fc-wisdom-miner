import json
import logging
import os
from typing import List

import requests
from flask import Flask, Response, render_template, request, stream_with_context

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import clean_html_file

app = Flask(__name__)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"
BASE_TMP_DIR = "tmp"

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_thread_dir(thread_url: str) -> str:
    thread_key = thread_url.rstrip("/").split("/")[-1]
    thread_dir = os.path.join(BASE_TMP_DIR, thread_key)
    os.makedirs(thread_dir, exist_ok=True)
    logger.debug(f"Thread directory resolved: {thread_dir}")
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
    logger.info(f"Preprocessing thread in {thread_dir}, force={force}")
    for filename in os.listdir(thread_dir):
        if filename.endswith(".html"):
            input_path = os.path.join(thread_dir, filename)
            output_path = os.path.join(thread_dir, filename.replace(".html", ".txt"))
            if force or not os.path.exists(output_path):
                logger.debug(f"Cleaning HTML file {input_path} -> {output_path}")
                clean_html_file(input_path, output_path)


def load_thread_text(thread_dir: str) -> str:
    texts: List[str] = []
    for fname in sorted(os.listdir(thread_dir)):
        if fname.endswith(".txt"):
            with open(os.path.join(thread_dir, fname), encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)


@app.route("/", methods=["GET", "POST"])
def index():
    threads = list_threads()
    return render_template("index.html", threads=threads)


@app.route("/ask", methods=["POST"])
def ask():
    base_url = request.form["url"].strip()
    user_prompt = request.form.get("prompt", "").strip()
    refresh = request.form.get("refresh", "") == "true"

    thread_dir = get_thread_dir(base_url)
    has_html = any(f.endswith(".html") for f in os.listdir(thread_dir))

    if refresh or not has_html:
        last_page = detect_last_page(base_url)
        fetch_forum_pages(base_url, start=1, end=last_page, save_dir=thread_dir)

    preprocess_thread(thread_dir, force=refresh)
    context = load_thread_text(thread_dir)

    system_prompt = (
        "You are an expert forum analyst. Based on the following anonymized forum content, "
        "answer the user's question with clear reasoning.\n\n"
        f"{context}\n\n"
        f"Question: {user_prompt}"
    )

    payload = {"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": True}
    logger.info("Sending prompt to Ollama...")

    def stream():
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as res:
            for line in res.iter_lines():
                if line:
                    try:
                        obj = json.loads(line.decode("utf-8"))
                        chunk = obj.get("response", "")
                        yield chunk
                    except Exception:
                        logger.exception("Streaming error")
                        yield "[Streaming Error]"

    return Response(stream_with_context(stream()), content_type="text/plain")


if __name__ == "__main__":
    logger.info("Starting app...")
    app.run(host="0.0.0.0", port=8080, debug=True)

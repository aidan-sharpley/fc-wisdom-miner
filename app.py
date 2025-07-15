import logging
import os
from typing import List, Optional

import requests
from flask import Flask, render_template, request

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import clean_html_file

app = Flask(__name__)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3.1:8b"
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


@app.route("/", methods=["GET", "POST"])
def index():
    threads = list_threads()
    response_text: Optional[str] = None
    selected_thread_url: Optional[str] = None

    if request.method == "POST":
        base_url = request.form["url"].strip()
        user_prompt = request.form.get("prompt", "").strip()
        refresh = request.form.get("refresh", "") == "true"
        selected_thread_url = base_url

        logger.info(f"User requested URL: {base_url} (refresh={refresh})")
        thread_dir = get_thread_dir(base_url)

        has_html = any(f.endswith(".html") for f in os.listdir(thread_dir))
        logger.debug(f"HTML files exist in thread dir: {has_html}")

        try:
            if refresh:
                if not has_html:
                    logger.info("No HTML found, fetching pages...")
                    last_page = detect_last_page(base_url)
                    logger.info(f"Detected last page: {last_page}")
                    fetch_forum_pages(
                        base_url, start=1, end=last_page, save_dir=thread_dir
                    )
                logger.info("Preprocessing all HTML files (force=True)...")
                preprocess_thread(thread_dir, force=True)
            else:
                if not has_html:
                    logger.info("No HTML found, fetching pages...")
                    last_page = detect_last_page(base_url)
                    logger.info(f"Detected last page: {last_page}")
                    fetch_forum_pages(
                        base_url, start=1, end=last_page, save_dir=thread_dir
                    )
                    preprocess_thread(thread_dir, force=False)

            context = load_thread_text(thread_dir)

            if user_prompt:
                logger.info(
                    f"Sending prompt to Ollama, prompt length: {len(user_prompt)}"
                )
                system_prompt = (
                    "You are an expert forum analyst. Based on the following anonymized forum content, "
                    "answer the user's question with clear reasoning.\n\n"
                    f"{context}\n\n"
                    f"Question: {user_prompt}"
                )

                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": system_prompt,
                    "stream": False,  # Change to False unless you handle streaming
                }
                res = requests.post(OLLAMA_API_URL, json=payload)

                if res.ok:
                    try:
                        data = res.json()
                        logger.debug(f"Ollama raw response JSON: {data}")
                        response_text = data.get("response")
                        if not response_text:
                            logger.warning(
                                "Ollama response key 'response' missing or empty"
                            )
                            response_text = "[No response content from Ollama]"
                    except Exception as e:
                        logger.error(f"Error decoding JSON from Ollama: {e}")
                        response_text = "[Error decoding Ollama response]"
                    logger.info("Received response from Ollama")
                else:
                    logger.error(f"Ollama API error: {res.status_code} {res.text}")
                    response_text = "[Error communicating with Ollama]"
            else:
                logger.debug("No user prompt provided, skipping Ollama request.")
                response_text = None

        except Exception as e:
            logger.exception("Exception occurred during request processing")
            response_text = f"[Error processing request: {e}]"

    return render_template(
        "index.html",
        threads=threads,
        response=response_text,
        selected_thread_url=selected_thread_url,
    )


if __name__ == "__main__":
    logger.info("Starting Flask app on 0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)

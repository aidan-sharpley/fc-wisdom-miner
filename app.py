import os
from typing import List, Optional

import requests
from flask import Flask, render_template, request

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import preprocess_all

app = Flask(__name__)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
TMP_DIR = "tmp"


def load_forum_corpus(thread_folder: str) -> str:
    texts: List[str] = []
    folder_path = os.path.join(TMP_DIR, thread_folder)
    if not os.path.isdir(folder_path):
        print(f"[load_forum_corpus] Folder not found: {folder_path}")
        return ""

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".txt"):
            file_path = os.path.join(folder_path, fname)
            with open(file_path, encoding="utf-8") as f:
                texts.append(f.read())
    corpus = "\n\n".join(texts)
    print(f"[load_forum_corpus] Loaded corpus length: {len(corpus)} chars")
    return corpus


def get_thread_folder(base_url: str) -> str:
    # Derive a folder name from the URL, e.g. the last segment "phase3-vaporizers.48407"
    thread_id = base_url.rstrip("/").split("/")[-1]
    return thread_id


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    response_text: str = ""
    error_msg: Optional[str] = None

    # List existing threads (folders in TMP_DIR)
    existing_threads: List[str] = []
    if os.path.exists(TMP_DIR):
        existing_threads = [
            d for d in os.listdir(TMP_DIR) if os.path.isdir(os.path.join(TMP_DIR, d))
        ]

    if request.method == "POST":
        base_url: str = request.form.get("url", "").strip()
        user_prompt: str = request.form.get("prompt", "").strip()
        selected_thread: str = request.form.get("thread_select", "").strip()

        print(f"[POST] base_url: {base_url}")
        print(f"[POST] user_prompt: {user_prompt}")
        print(f"[POST] selected_thread: {selected_thread}")

        # Choose which thread to use: selected existing or new from URL
        if selected_thread:
            thread_folder = selected_thread
            print(f"[POST] Using existing thread folder: {thread_folder}")
        elif base_url:
            thread_folder = get_thread_folder(base_url)
            print(f"[POST] Using new thread folder from URL: {thread_folder}")
        else:
            error_msg = "Please provide a thread URL or select an existing thread."
            return render_template(
                "index.html",
                response=response_text,
                error=error_msg,
                existing_threads=existing_threads,
            )

        folder_path = os.path.join(TMP_DIR, thread_folder)
        os.makedirs(folder_path, exist_ok=True)

        # Check if we already have .txt files in the thread folder
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        try:
            if not txt_files:
                # Fetch pages and preprocess only if no existing data
                print(
                    f"[POST] No existing txt files found in {folder_path}, fetching and preprocessing..."
                )
                if not base_url:
                    error_msg = "Base URL is required to fetch new thread data."
                    return render_template(
                        "index.html",
                        response=response_text,
                        error=error_msg,
                        existing_threads=existing_threads,
                    )

                # Detect last page to fetch
                last_page = detect_last_page(base_url)
                print(f"[POST] Detected last page: {last_page}")

                # Fetch forum pages into thread folder
                fetch_forum_pages(
                    base_url, start=1, end=last_page, save_folder=folder_path
                )
                preprocess_all(thread_folder)
            else:
                print(
                    f"[POST] Found existing txt files in {folder_path}, skipping fetch/preprocess."
                )

            # Load corpus from thread folder
            context = load_forum_corpus(thread_folder)
            if not context.strip():
                error_msg = "Failed to load content from forum thread."
                return render_template(
                    "index.html",
                    response=response_text,
                    error=error_msg,
                    existing_threads=existing_threads,
                )

            system_prompt = (
                "You are an expert forum analyst. Based on the following anonymized forum content, "
                "answer the user's question with clear reasoning.\n\n"
                f"{context}\n\n"
                f"Question: {user_prompt}"
            )

            payload = {"model": OLLAMA_MODEL, "prompt": system_prompt, "stream": False}
            print(f"[POST] Sending request to Ollama API: {OLLAMA_API_URL}")
            res = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
            res.raise_for_status()

            response_json = res.json()
            response_text = response_json.get("response", "[No response from model]")
            print(f"[POST] Ollama response length: {len(response_text)} chars")

        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with Ollama API: {e}"
            print(f"[ERROR] {error_msg}")

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            print(f"[ERROR] {error_msg}")

    return render_template(
        "index.html",
        response=response_text,
        error=error_msg,
        existing_threads=existing_threads,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

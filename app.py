import os
from typing import Any, Dict, Optional

import requests
from flask import Flask, redirect, render_template, request, url_for

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import load_thread_text, preprocess_thread

app = Flask(__name__)

TMP_DIR = "tmp"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


def list_threads() -> list[str]:
    if not os.path.exists(TMP_DIR):
        return []
    return sorted(
        [d for d in os.listdir(TMP_DIR) if os.path.isdir(os.path.join(TMP_DIR, d))]
    )


def call_ollama(prompt: str) -> str:
    payload: Dict[str, Any] = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        res = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "[No response]")
    except Exception as e:
        return f"[Error communicating with Ollama: {e}]"


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    threads = list_threads()
    response_text: str = ""
    selected_thread: Optional[str] = None
    question: str = ""

    if request.method == "POST":
        action = request.form.get("action")
        if action == "add_thread":
            base_url = request.form.get("new_url", "").strip()
            if base_url:
                # Extract thread id from url (last part after .)
                # Example: https://fuckcombustion.com/threads/phase3-vaporizers.48407
                thread_id = base_url.rstrip("/").split(".")[-1]
                thread_dir = os.path.join(TMP_DIR, thread_id)
                if not os.path.exists(thread_dir):
                    os.makedirs(thread_dir, exist_ok=True)
                    last_page = detect_last_page(base_url)
                    fetch_forum_pages(base_url, thread_id, 1, last_page)
                    preprocess_thread(thread_dir)
                return redirect(url_for("index", thread=thread_id))
        elif action == "ask_question":
            selected_thread = request.form.get("thread_select")
            question = request.form.get("prompt", "").strip()
            if selected_thread and question:
                thread_dir = os.path.join(TMP_DIR, selected_thread)
                if os.path.exists(thread_dir):
                    context = load_thread_text(thread_dir)
                    system_prompt = (
                        "You are an expert forum analyst. Based on the following anonymized forum content, "
                        "answer the user's question with clear reasoning.\n\n"
                        f"{context}\n\n"
                        f"Question: {question}"
                    )
                    response_text = call_ollama(system_prompt)

    else:
        selected_thread = request.args.get("thread")

    return render_template(
        "index.html",
        threads=threads,
        response=response_text,
        selected_thread=selected_thread,
        question=question,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

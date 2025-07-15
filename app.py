import html
import json
import logging
import os
import re
from typing import List

import requests
from flask import Flask, Response, render_template, request

from fetch_forum import detect_last_page, fetch_forum_pages
from preprocess import clean_html_file

app = Flask(__name__)
app.secret_key = "replace-with-a-secure-random-secret"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:3.8b"
# OLLAMA_MODEL = "deepseek-r1:7b" okay
# OLLAMA_MODEL = "llama3.1:8b" unusable
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
    # Unescape HTML entities
    text = html.unescape(raw)

    # Remove all quote blocks like:
    #    username said:\n content \nClick to expand...
    text = re.sub(
        r"(?:^|\n)\s*\w+\s+said:\s*\n.*?(?:Click to expand\.\.\.)?",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove remaining "Click to expand..." anywhere
    text = re.sub(r"Click to expand\.\.\.", "", text)

    # Normalize spacing
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

    system_prompt = f"""
        You are an expert forum analyst helping a user explore a forum thread that contains multiple posts per page and multiple pages within the thread.

        The user will ask questions about the thread, which consists of posts with metadata in JSON format: each post has a page number, ISO date, and textual content.
        
        Use this metadata format for posts within the combined context: '{{"page": 1, "date": "2020-03-12T23:22:56-0400", "content": "If when you choose to sell any, you'll need to let us know in advance in order to set you up as a commercial account, ok? Great work!"}}'

        By default, provide clear, conversational, helpful answers that address the user's question directly without excessive explanation.

        If the user asks for detailed reasoning or evidence, then provide supporting post metadata including the post date and a direct link to the post on the forum. Construct post links by appending the post ID (if available) to the base URL: 
        https://fuckcombustion.com/threads/phase3-vaporizers.48407/post-{{post_id}}.

        Use the anonymized content for context and keep answers friendly and expert.
        
        All responses should be in English, even if the original posts are in another language.
        
        If the user asks for a summary, provide a concise overview of the thread's main points and discussions.
        
        If the user asks for a list of posts, provide a numbered list with brief summaries of each post.
        
        If the user asks for a specific post, provide the post's content along with its metadata.
        
        If the user asks for a specific page, provide a summary of that page's posts.
        
        All responses should be concise, relevant, and directly address the user's question.
        ---
        Context is provided below. Use it to answer the user's question, but do not repeat the context verbatim in your response.
        <<{context}>>
        ---
        Everything below this line and between the "<<>>" is the user's question. Do not include it in your response.
        <<{prompt}>>
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


if __name__ == "__main__":
    logger.info("Launching app at http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080, debug=True)

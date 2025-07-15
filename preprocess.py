import os
import re
from typing import List

from bs4 import BeautifulSoup


def clean_html_file(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    for selector in [".username", ".user-info", ".post-author", ".user"]:
        for tag in soup.select(selector):
            tag.decompose()

    posts = soup.select(".post")
    if not posts:
        posts = soup.select("article, div.postbody, .content, p")

    content = "\n\n".join(post.get_text(strip=True) for post in posts)

    content = re.sub(
        r"\bUser\d+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
        "[USER]",
        content,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def preprocess_thread(thread_dir: str) -> None:
    for filename in os.listdir(thread_dir):
        if filename.endswith(".html"):
            input_path = os.path.join(thread_dir, filename)
            output_path = os.path.join(thread_dir, filename.replace(".html", ".txt"))
            if not os.path.exists(output_path):
                clean_html_file(input_path, output_path)


def load_thread_text(thread_dir: str) -> str:
    texts: List[str] = []
    for fname in sorted(os.listdir(thread_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(thread_dir, fname)
            with open(path, encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)

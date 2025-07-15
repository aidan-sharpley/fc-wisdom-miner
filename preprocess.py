import os
import re
from typing import Optional

from bs4 import BeautifulSoup


def clean_html_file(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Remove usernames, user info etc
    for selector in [".username", ".user-info", ".post-author", ".user"]:
        for tag in soup.select(selector):
            tag.decompose()

    posts = soup.select(".post")
    if not posts:
        posts = soup.select("article, div.postbody, .content, p")

    content = "\n\n".join(post.get_text(strip=True) for post in posts)

    # Anonymize user patterns
    content = re.sub(
        r"\bUser\d+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
        "[USER]",
        content,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def preprocess_all(directory: Optional[str] = None, force: bool = False) -> None:
    """
    Preprocess all html files in directory (or tmp if None).
    If force=True, overwrite .txt files.
    """
    if directory is None:
        directory = "tmp"
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename.replace(".html", ".txt"))
            if force or not os.path.exists(output_path):
                clean_html_file(input_path, output_path)

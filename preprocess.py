import os
import re
from typing import Optional

from bs4 import BeautifulSoup


def clean_html_file(input_path: str, output_path: str) -> None:
    """
    Parse and anonymize forum HTML page, extracting text posts.
    Remove user info, usernames, emails replaced by [USER].
    """
    with open(input_path, encoding="utf-8") as f:
        soup: BeautifulSoup = BeautifulSoup(f, "lxml")

    for selector in [".username", ".user-info", ".post-author", ".user"]:
        for tag in soup.select(selector):
            tag.decompose()

    posts: list = soup.select(".post")
    if not posts:
        posts = soup.select("article, div.postbody, .content, p")

    content: str = "\n\n".join(post.get_text(strip=True) for post in posts)

    # Replace usernames and emails with [USER]
    content = re.sub(
        r"\bUser\d+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
        "[USER]",
        content,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def preprocess_all(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """
    Process all .html files in input_dir (default 'tmp'),
    save processed .txt files to output_dir (default same as input_dir).
    """
    if input_dir is None:
        input_dir = "tmp"
    if output_dir is None:
        output_dir = input_dir

    for filename in os.listdir(input_dir):
        if filename.endswith(".html"):
            input_path: str = os.path.join(input_dir, filename)
            output_path: str = os.path.join(
                output_dir, filename.replace(".html", ".txt")
            )
            clean_html_file(input_path, output_path)

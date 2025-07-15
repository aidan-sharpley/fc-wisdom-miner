# preprocess.py
import json
import os
import re
from typing import Optional

from bs4 import BeautifulSoup

TMP_DIR = "tmp"


def extract_post_date(post_tag) -> Optional[str]:
    """
    Extract a post date string from a post HTML tag.
    Adjust selectors based on forum HTML structure.
    """
    # Common date selectors to try
    date_tag = post_tag.select_one(".post-date, .date, time")
    if date_tag:
        # Check for datetime attribute first
        if date_tag.has_attr("datetime"):
            return date_tag["datetime"]
        return date_tag.get_text(strip=True)
    return None


def clean_html_file(input_path: str, output_path: str) -> None:
    with open(input_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    # Remove username and user info elements to anonymize
    for selector in [".username", ".user-info", ".post-author", ".user"]:
        for tag in soup.select(selector):
            tag.decompose()

    posts = soup.select(".post")
    if not posts:
        posts = soup.select("article, div.postbody, .content, p")

    results = []

    for post in posts:
        content = post.get_text(strip=True)

        # Anonymize usernames/emails inside post content
        content = re.sub(
            r"\bUser\d+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}",
            "[USER]",
            content,
        )

        date_str = extract_post_date(post)

        results.append(
            {
                "date": date_str or "",
                "content": content,
            }
        )

    # Write out as JSON lines, one post per line
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_thread_text(thread_dir: str) -> str:
    """
    Load all .txt files in the thread directory,
    parse JSON lines, and combine posts with dates.
    """
    texts = []
    for fname in sorted(os.listdir(thread_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(thread_dir, fname)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        post = json.loads(line)
                        date = post.get("date", "")
                        content = post.get("content", "")
                        if date:
                            text = f"[{date}]\n{content}"
                        else:
                            text = content
                        texts.append(text)
                    except json.JSONDecodeError:
                        # If line is not JSON, skip it
                        continue
    return "\n\n".join(texts)

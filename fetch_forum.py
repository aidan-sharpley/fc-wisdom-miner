import os
from typing import Optional

import requests
from bs4 import BeautifulSoup

TMP_DIR = "tmp"


def fetch_forum_pages(
    base_url: str, start: int = 1, end: int = 3, save_dir: Optional[str] = None
) -> None:
    """
    Downloads pages from base_url which should NOT include the /page-N part.
    Saves HTML files into save_dir or TMP_DIR if not given.
    Does NOT redownload pages if .html exists.
    """
    if save_dir is None:
        save_dir = TMP_DIR
    os.makedirs(save_dir, exist_ok=True)

    for i in range(start, end + 1):
        filename = f"page_{i}.html"
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            print(f"Skipping download, already have: {filename}")
            continue

        url = f"{base_url}/page-{i}"
        print(f"Fetching: {url}")
        res = requests.get(url)
        if res.status_code == 404:
            print(f"Page {i} not found (404). Stopping.")
            break
        res.raise_for_status()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(res.text)


def detect_last_page(base_url: str) -> int:
    url = f"{base_url}/page-1"
    res = requests.get(url)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "lxml")

    pages = []
    for link in soup.select(".pageNav-page"):
        try:
            p = int(link.get_text())
            pages.append(p)
        except Exception:
            continue
    if pages:
        return max(pages)
    return 1

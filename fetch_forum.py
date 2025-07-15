import os
from typing import Optional

import requests
from bs4 import BeautifulSoup


def fetch_forum_pages(
    base_url: str,
    start: int = 1,
    end: int = 3,
    save_dir: Optional[str] = None,
) -> None:
    """
    Download forum pages from base_url (without /page-N).
    Save HTML files into save_dir or 'tmp' by default.
    Stop if page 404 encountered.
    """
    if save_dir is None:
        save_dir = "tmp"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(start, end + 1):
        filename: str = f"page_{i}.html"
        filepath: str = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            print(f"Skipping existing page {i} at {filepath}")
            continue

        url: str = f"{base_url}/page-{i}"
        print(f"Fetching: {url}")
        res: requests.Response = requests.get(url)
        if res.status_code == 404:
            print(f"Page {i} not found (404). Stopping.")
            break
        res.raise_for_status()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(res.text)


def detect_last_page(base_url: str) -> int:
    """
    Detect the last page number of the thread by parsing pagination.
    Returns 1 if no pagination found.
    """
    url: str = f"{base_url}/page-1"
    res: requests.Response = requests.get(url)
    res.raise_for_status()

    soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")

    pages: list[int] = []
    for link in soup.select(".pageNav-page"):
        try:
            p: int = int(link.get_text())
            pages.append(p)
        except ValueError:
            continue
    if pages:
        return max(pages)
    return 1

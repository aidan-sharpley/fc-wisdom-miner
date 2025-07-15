import os

import requests
from bs4 import BeautifulSoup


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
    return max(pages) if pages else 1


def fetch_forum_pages(
    base_url: str, thread_id: str, start: int, end: int, base_tmp_dir: str = "tmp"
) -> None:
    """
    Fetch forum pages for a thread, save each page as HTML in tmp/thread_id/page_N.html.
    """
    thread_dir = os.path.join(base_tmp_dir, thread_id)
    os.makedirs(thread_dir, exist_ok=True)

    for i in range(start, end + 1):
        filename = os.path.join(thread_dir, f"page_{i}.html")
        if os.path.exists(filename):
            print(f"Page {i} already exists, skipping download.")
            continue
        url = f"{base_url}/page-{i}"
        print(f"Fetching: {url}")
        res = requests.get(url)
        if res.status_code == 404:
            print(f"Page {i} not found (404). Stopping.")
            break
        res.raise_for_status()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(res.text)

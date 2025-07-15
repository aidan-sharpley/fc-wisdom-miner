import os
import time

import requests
from bs4 import BeautifulSoup


def detect_last_page(base_url: str) -> int:
    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, "html.parser")

    last_link = soup.select_one("a.pageNav-jump--last")
    if last_link and last_link.text.isdigit():
        return int(last_link.text)

    page_links = soup.select("a.pageNav-page")
    nums = [int(link.text) for link in page_links if link.text.isdigit()]
    return max(nums) if nums else 1


def fetch_forum_pages(base_url: str, start: int, end: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for i in range(start, end + 1):
        page_url = base_url if i == 1 else f"{base_url}page-{i}"
        output_path = os.path.join(save_dir, f"page{i}.html")

        if os.path.exists(output_path):
            continue

        print(f"Fetching page {i}...")
        res = requests.get(page_url)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(res.text)

        time.sleep(1)

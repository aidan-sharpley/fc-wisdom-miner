import logging
import os
import re
import time

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def detect_last_page(base_url: str) -> int:
    res = requests.get(base_url)
    if res.status_code != 200:
        logger.error(f"Error: Received status code {res.status_code}")
        return 1

    soup = BeautifulSoup(res.text, "html.parser")
    logger.info("Base page pulled successfully.")

    last_page = 1  # Initialize with 1

    # First, try to find the 'max' attribute in the page jump input, if available.
    # This is often the most direct way to get the true last page.
    page_jump_input = soup.select_one('.js-pageJumpPage[type="number"]')
    if page_jump_input and "max" in page_jump_input.attrs:
        try:
            max_page_from_input = int(page_jump_input["max"])
            if max_page_from_input > last_page:
                last_page = max_page_from_input
                logger.info(f"Detected last page from page jump input: {last_page}")
                return last_page  # If found, this is highly reliable, so return immediately.
        except ValueError:
            pass  # Ignore if max is not a valid number

    # Fallback: Iterate through all page number links and find the highest one.
    # This handles cases where there's no explicit 'last' link or max attribute.
    page_links = soup.select(".pageNav-page a")  # Select <a> tags within .pageNav-page
    for link in page_links:
        try:
            # Try to get number from href (e.g., page-227)
            href = link.get("href")
            if href:
                match = re.search(r"page-(\d+)", href)
                if match:
                    page_num_from_href = int(match.group(1))
                    if page_num_from_href > last_page:
                        last_page = page_num_from_href

            # Also try to get number from text content (e.g., 227)
            link_text = link.text.strip()
            if link_text.isdigit():
                page_num_from_text = int(link_text)
                if page_num_from_text > last_page:
                    last_page = page_num_from_text
        except ValueError:
            continue  # Ignore links that don't contain valid numbers

    logger.info(f"Detected last page: {last_page}")
    return last_page


def fetch_forum_pages(base_url: str, start: int, end: int, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for i in range(start, end + 1):
        page_url = base_url if i == 1 else f"{base_url}page-{i}"
        output_path = os.path.join(save_dir, f"page{i}.html")

        if os.path.exists(output_path):
            continue

        logger.info(f"Fetching page {i}...")
        res = requests.get(page_url)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(res.text)

        time.sleep(1)

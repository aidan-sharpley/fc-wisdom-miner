import logging
from urllib.parse import urljoin

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_date(post_element) -> str:
    time_tag = post_element.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        return time_tag["datetime"]
    return "unknown-date"


def extract_content(post_element) -> str:
    # Pulls post content from XenForo bbWrapper
    content_div = post_element.select_one(".message-main .bbWrapper")
    return content_div.get_text(separator="\n").strip() if content_div else ""


def extract_post_url(post_element: BeautifulSoup, canonical_base: str) -> str:
    """Build a full permalink from relative href or fallback to post ID."""
    permalink_element = post_element.select_one(".message-attribution-main a")
    if permalink_element and permalink_element.get("href"):
        return urljoin(canonical_base, permalink_element["href"])

    # Fallback: construct from post ID
    if post_element.has_attr("id") and post_element["id"].startswith("post-"):
        post_id = post_element["id"].split("-")[1]
        return f"{canonical_base}post-{post_id}/"

    return f"{canonical_base}unknown-post"


def extract_canonical_url(soup: BeautifulSoup) -> str:
    """Grabs the base thread URL from the <link rel='canonical'> tag."""
    canonical_link = soup.select_one("head link[rel='canonical']")
    return (
        canonical_link["href"].rstrip("/") + "/"
        if canonical_link
        else "unknown-thread/"
    )

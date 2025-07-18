import json

from bs4 import BeautifulSoup


def extract_date(post_element) -> str:
    time_tag = post_element.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        return time_tag["datetime"]
    return "unknown-date"


def extract_content(post_element) -> str:
    # Extracts from the bbWrapper class inside message-body
    content_div = post_element.select_one(".message-body .bbWrapper")
    return content_div.get_text(separator="\n").strip() if content_div else ""


def extract_post_url(post_element: BeautifulSoup) -> str:
    """Extracts the direct URL of a post."""
    # Common selectors for permalinks or post IDs
    permalink_element = post_element.select_one(
        ".permalink, .message-attribution-main a"
    )
    if permalink_element and permalink_element.get("href"):
        # Construct full URL if it's a relative path
        relative_url = permalink_element["href"]
        # You might need more sophisticated URL joining if base_url is not just the domain
        # For simplicity, assuming base_url is something like "http://example.com"
        from urllib.parse import urljoin

        return urljoin(relative_url.lstrip("/"))
    # Fallback: construct from post ID if available
    if "id" in post_element.attrs and post_element["id"].startswith("post-"):
        post_id = post_element["id"].split("-")[1]
        return post_id
        # This assumes your forum URLs follow a pattern like base_url/post-ID
        # You'll need to adapt this based on your specific forum's URL structure
        # If your thread URLs are structured like /threads/thread_title.ID/, then this might be more complex.
        # It might be easier to use the href from the permalink as above.
        # For a XenForo forum, a permalink often looks like /threads/thread_title.ID/post-POSTID
        # You might need the thread_key from app.py to construct this.
        # For now, relying on explicit permalink element.
        pass
    return "Unknown URL"


def clean_html_file(input_path: str, output_path: str, page_num: int) -> None:
    with open(input_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Each post container has class="message message--post ..."
    post_elements = soup.select("article.message")

    posts = []
    for post_el in post_elements:
        date_str = extract_date(post_el)
        content_str = extract_content(post_el)
        if content_str.strip():  # Only include posts with actual content
            posts.append(
                {
                    "page": page_num,
                    "date": date_str,
                    "content": content_str,
                    "url": extract_post_url(post_el),
                }
            )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for post in posts:
            out_f.write(json.dumps(post) + "\n")

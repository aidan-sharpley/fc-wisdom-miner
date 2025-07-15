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
                }
            )

    with open(output_path, "w", encoding="utf-8") as out_f:
        for post in posts:
            out_f.write(json.dumps(post) + "\n")

import json

from bs4 import BeautifulSoup


def extract_date(post_element) -> str:
    # TODO: Implement extraction of ISO8601 date string from post_element
    # Example placeholder:
    date_tag = post_element.find("time")
    if date_tag and date_tag.has_attr("datetime"):
        return date_tag["datetime"]
    # Fallback or parsing alternative if needed
    return "unknown-date"


def extract_content(post_element) -> str:
    # TODO: Extract and clean post content text from post_element
    # Example placeholder:
    content_div = post_element.find("div", class_="post-content")
    return content_div.get_text(separator="\n").strip() if content_div else ""


def clean_html_file(input_path: str, output_path: str, page_num: int) -> None:
    with open(input_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Replace with your site's actual post container selector
    post_elements = soup.find_all("div", class_="post")

    posts = []
    for post_el in post_elements:
        date_str = extract_date(post_el)
        content_str = extract_content(post_el)
        post = {
            "page": page_num,
            "date": date_str,
            "content": content_str,
        }
        posts.append(post)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for post in posts:
            out_f.write(json.dumps(post) + "\n")

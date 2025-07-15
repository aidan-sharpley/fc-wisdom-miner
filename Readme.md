# Forum Thread Q&A App

This Flask web app lets you fetch, preprocess, and query multi-page forum threads using a local Ollama LLM (`deepseek-r1:1.5b` model). It downloads forum pages, anonymizes user info, and allows you to ask questions about the thread content.

---

## Features

- Fetches multi-page forum threads, skipping downloads if HTML already exists
- Cleans and anonymizes downloaded HTML to plain text for LLM context
- Supports selecting previously downloaded threads or adding new thread URLs
- Allows refreshing thread data (redownloads and reprocesses)
- Queries Ollama LLM with thread context and user question

---

## Requirements

- Python 3.9+
- [Flask](https://flask.palletsprojects.com/)
- [Requests](https://docs.python-requests.org/)
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)
- [Ollama](https://ollama.com/) running locally with model `deepseek-r1:1.5b`

---

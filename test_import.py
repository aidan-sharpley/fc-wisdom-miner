#!/usr/bin/env python3
"""Test script to isolate import issues."""

print("Starting imports...")

try:
    print("1. Basic imports...")
    import logging
    import os
    import threading
    import time
    import weakref
    from typing import Dict, List, Optional
    print("✓ Basic imports successful")

    print("2. Flask imports...")
    from flask import Flask, Response, jsonify, render_template, request
    print("✓ Flask imports successful")

    print("3. Analytics imports...")
    from analytics.query_analytics import ConversationalQueryProcessor
    from analytics.thread_analyzer import ThreadAnalyzer
    print("✓ Analytics imports successful")

    print("4. Config imports...")
    from config.settings import (
        BASE_TMP_DIR,
        EMBEDDING_BATCH_SIZE,
        FEATURES,
        MAX_LOGGED_PROMPT_LENGTH,
        MAX_WORKERS,
        OLLAMA_BASE_URL,
        OLLAMA_CHAT_MODEL,
        OLLAMA_EMBED_MODEL,
        QUERY_PROCESSOR_CACHE_SIZE,
        THREADS_DIR,
    )
    print("✓ Config imports successful")

    print("5. Processing imports...")
    from processing.thread_processor import ThreadProcessor
    from search.query_processor import QueryProcessor
    print("✓ Processing imports successful")

    print("6. Utils imports...")
    from utils.file_utils import get_thread_dir, safe_read_json
    from utils.helpers import normalize_url
    from utils.memory_optimizer import MemoryMonitor, get_memory_status
    from utils.question_history import get_question_history
    from utils.shared_data_manager import get_data_manager
    print("✓ Utils imports successful")

    print("7. Creating instances...")
    thread_processor = ThreadProcessor()
    memory_monitor = MemoryMonitor()
    print("✓ Instance creation successful")

    print("8. Setting up logging...")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(BASE_TMP_DIR, 'app.log'), mode='a'),
        ],
    )
    print("✓ Logging setup successful")

    print("9. Creating Flask app...")
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY', 'forum-wisdom-miner-secret-key')
    print("✓ Flask app creation successful")

    print("\n🎉 All imports and setup completed successfully!")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
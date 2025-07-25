#!/usr/bin/env python3
"""Test the main execution path of app.py step by step."""

import os
import time
import logging

# Import everything from app
from app import *

def test_main_execution():
    print("Testing main execution path...")
    
    try:
        print("1. Creating directories...")
        os.makedirs(BASE_TMP_DIR, exist_ok=True)
        os.makedirs(THREADS_DIR, exist_ok=True)
        print("✓ Directories created")

        print("2. Testing log file creation...")
        log_file = os.path.join(BASE_TMP_DIR, 'app.log')
        with open(log_file, 'a') as f:
            f.write(f'\n--- Test run at {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')
        print("✓ Log file created")

        print("3. Testing logger setup...")
        logger = logging.getLogger(__name__)
        logger.info('Test log message')
        print("✓ Logger working")

        print("4. Testing list_available_threads...")
        existing_threads = list_available_threads()
        print(f"✓ Found {len(existing_threads)} existing threads")

        print("5. Testing Flask app configuration...")
        debug_mode = os.environ.get('FLASK_ENV') == 'development'
        port = int(os.environ.get('PORT', 8080))
        print(f"✓ Flask config: debug={debug_mode}, port={port}")

        print("\n🎉 All main execution steps completed successfully!")
        print("The issue might be in Flask's app.run() method...")

    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_main_execution()
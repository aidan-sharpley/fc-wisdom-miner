"""
Question History Manager - Per-thread question reuse functionality.

Manages the last 5 distinct questions asked per thread for easy reuse.
Uses in-memory storage with disk backup for persistence across restarts.
"""

import json
import os
import threading
from collections import deque
from typing import Dict, List, Optional

from config.settings import BASE_TMP_DIR


class QuestionHistory:
    """Thread-safe question history manager with per-thread storage."""
    
    def __init__(self, max_questions: int = 5):
        self.max_questions = max_questions
        self._history: Dict[str, deque] = {}
        self._lock = threading.RLock()
        self._history_file = os.path.join(BASE_TMP_DIR, 'question_history.json')
        
        # Load existing history on startup
        self._load_history()
    
    def add_question(self, thread_key: str, question: str) -> None:
        """Add a question to the history for a specific thread.
        
        Args:
            thread_key: Thread identifier
            question: User question to add to history
        """
        if not thread_key or not question or not question.strip():
            return
        
        question = question.strip()
        
        with self._lock:
            # Initialize thread history if not exists
            if thread_key not in self._history:
                self._history[thread_key] = deque(maxlen=self.max_questions)
            
            thread_history = self._history[thread_key]
            
            # Remove duplicate if exists (move to front)
            if question in thread_history:
                thread_history.remove(question)
            
            # Add to front (most recent)
            thread_history.appendleft(question)
            
            # Persist to disk
            self._save_history()
    
    def get_questions(self, thread_key: str) -> List[str]:
        """Get question history for a specific thread.
        
        Args:
            thread_key: Thread identifier
            
        Returns:
            List of recent questions (most recent first)
        """
        if not thread_key:
            return []
        
        with self._lock:
            if thread_key not in self._history:
                return []
            
            return list(self._history[thread_key])
    
    def clear_thread_history(self, thread_key: str) -> None:
        """Clear question history for a specific thread.
        
        Args:
            thread_key: Thread identifier
        """
        if not thread_key:
            return
        
        with self._lock:
            if thread_key in self._history:
                del self._history[thread_key]
                self._save_history()
    
    def get_all_threads(self) -> List[str]:
        """Get list of all threads with question history.
        
        Returns:
            List of thread keys that have question history
        """
        with self._lock:
            return list(self._history.keys())
    
    def get_total_questions(self) -> int:
        """Get total number of questions across all threads.
        
        Returns:
            Total question count
        """
        with self._lock:
            return sum(len(history) for history in self._history.values())
    
    def _load_history(self) -> None:
        """Load question history from disk."""
        try:
            if os.path.exists(self._history_file):
                with open(self._history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert lists back to deques with max length
                for thread_key, questions in data.items():
                    self._history[thread_key] = deque(questions, maxlen=self.max_questions)
                    
        except Exception as e:
            # Log error but don't fail - start with empty history
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f'Failed to load question history: {e}')
            self._history = {}
    
    def _save_history(self) -> None:
        """Save question history to disk."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._history_file), exist_ok=True)
            
            # Convert deques to lists for JSON serialization
            data = {
                thread_key: list(history) 
                for thread_key, history in self._history.items()
            }
            
            # Write atomically using temporary file
            temp_file = self._history_file + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            if os.path.exists(self._history_file):
                os.replace(temp_file, self._history_file)
            else:
                os.rename(temp_file, self._history_file)
                
        except Exception as e:
            # Log error but don't fail application
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f'Failed to save question history: {e}')


# Global instance
_question_history = None
_history_lock = threading.Lock()


def get_question_history() -> QuestionHistory:
    """Get the global question history instance (singleton pattern)."""
    global _question_history
    
    if _question_history is None:
        with _history_lock:
            if _question_history is None:
                _question_history = QuestionHistory()
    
    return _question_history
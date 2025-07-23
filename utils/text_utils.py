"""
Text processing utilities for Forum Wisdom Miner.

This module contains functions for cleaning, normalizing, and processing
text content from forum posts.
"""

import logging
import re
from typing import List, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def clean_post_content(raw: str) -> str:
    """Clean HTML and forum-specific formatting from post content.
    
    This function removes HTML tags, forum-specific patterns, and normalizes
    whitespace to produce clean text suitable for embedding and analysis.
    
    Args:
        raw: Raw HTML content from forum post
        
    Returns:
        Cleaned text content
    """
    if not raw:
        return ""

    try:
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script and style elements completely
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Remove quote blocks to avoid duplication
        for quote in soup.select('.quote, .quotebox, blockquote'):
            quote.decompose()

        # Remove signature blocks
        for sig in soup.select('.signature, .sig, .postbit_signature'):
            sig.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up forum-specific patterns
        text = _clean_forum_patterns(text)
        
        # Normalize whitespace
        text = _normalize_whitespace(text)
        
        return text.strip()
        
    except Exception as e:
        logger.warning(f"Error cleaning post content: {e}")
        return str(raw)[:1000]  # Fallback to truncated raw content


def _clean_forum_patterns(text: str) -> str:
    """Remove forum-specific text patterns."""
    # Remove quote attribution patterns
    text = re.sub(r"(?:^|\n)\s*\w+ said:\s*(Click to expand\.{3})?", "", text)
    text = re.sub(r"Click to expand\.{3}", "", text)
    text = re.sub(r"Quote:\s*", "", text)
    text = re.sub(r"Originally posted by.*?:", "", text)
    
    # Remove edit notices
    text = re.sub(r"Last edited by.*?;", "", text)
    text = re.sub(r"Edit:.*?$", "", text, flags=re.MULTILINE)
    
    # Remove reaction patterns
    text = re.sub(r"Like\s*\|\s*Dislike", "", text)
    text = re.sub(r"Thanks\s*\|\s*Reply", "", text)
    
    # Remove navigation elements
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Jump to page:", "", text)
    
    return text


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Convert multiple spaces/tabs to single space
    text = re.sub(r"[\t ]+", " ", text)
    
    # Limit consecutive newlines to maximum of 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Clean up mixed whitespace around newlines
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    
    return text


def extract_keywords(text: str, min_length: int = 3, max_count: int = 20) -> List[str]:
    """Extract potential keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum length for keywords
        max_count: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Simple keyword extraction - remove common stop words and extract meaningful terms
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Filter and collect keywords
    keywords = []
    for word in words:
        if (len(word) >= min_length and 
            word not in stop_words and
            word not in keywords):
            keywords.append(word)
            
        if len(keywords) >= max_count:
            break
    
    return keywords


def detect_language(text: str) -> str:
    """Simple language detection based on common patterns.
    
    Args:
        text: Input text
        
    Returns:
        Language code ('en', 'es', 'fr', etc.) or 'unknown'
    """
    if not text or len(text) < 50:
        return 'unknown'
    
    text_lower = text.lower()
    
    # Simple pattern-based detection
    english_indicators = ['the ', ' and ', ' of ', ' to ', ' in ', ' is ', ' that ']
    spanish_indicators = ['el ', ' y ', ' de ', ' en ', ' es ', ' que ', ' la ']
    french_indicators = ['le ', ' et ', ' de ', ' à ', ' est ', ' que ', ' la ']
    
    en_count = sum(1 for pattern in english_indicators if pattern in text_lower)
    es_count = sum(1 for pattern in spanish_indicators if pattern in text_lower)
    fr_count = sum(1 for pattern in french_indicators if pattern in text_lower)
    
    max_count = max(en_count, es_count, fr_count)
    
    if max_count >= 3:  # Require at least 3 matches
        if en_count == max_count:
            return 'en'
        elif es_count == max_count:
            return 'es'
        elif fr_count == max_count:
            return 'fr'
    
    return 'unknown'


def calculate_readability_score(text: str) -> float:
    """Calculate a simple readability score for text.
    
    Args:
        text: Input text
        
    Returns:
        Readability score (higher = more readable)
    """
    if not text:
        return 0.0
    
    # Count sentences, words, and syllables (approximate)
    sentences = len(re.findall(r'[.!?]+', text))
    words = len(re.findall(r'\b\w+\b', text))
    
    if sentences == 0 or words == 0:
        return 0.0
    
    # Simple approximation of syllables
    syllables = 0
    for word in re.findall(r'\b\w+\b', text.lower()):
        # Count vowel groups as syllables
        syllable_count = len(re.findall(r'[aeiouy]+', word))
        syllables += max(1, syllable_count)  # Every word has at least 1 syllable
    
    # Simplified Flesch Reading Ease formula
    avg_sentence_length = words / sentences
    avg_syllables_per_word = syllables / words
    
    score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    
    # Normalize to 0-100 range
    return max(0.0, min(100.0, score))


def extract_sentences(text: str, max_sentences: int = 10) -> List[str]:
    """Extract individual sentences from text.
    
    Args:
        text: Input text
        max_sentences: Maximum number of sentences to return
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Split on sentence endings
    sentences = re.split(r'[.!?]+', text)
    
    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Filter out very short fragments
            cleaned_sentences.append(sentence)
            
        if len(cleaned_sentences) >= max_sentences:
            break
    
    return cleaned_sentences


def get_text_statistics(text: str) -> dict:
    """Get comprehensive statistics about text content.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'characters': 0,
            'words': 0,
            'sentences': 0,
            'paragraphs': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'readability_score': 0.0,
            'language': 'unknown'
        }
    
    characters = len(text)
    words = len(re.findall(r'\b\w+\b', text))
    sentences = len(re.findall(r'[.!?]+', text))
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    
    # Calculate averages
    avg_word_length = sum(len(word) for word in re.findall(r'\b\w+\b', text)) / max(1, words)
    avg_sentence_length = words / max(1, sentences)
    
    return {
        'characters': characters,
        'words': words,
        'sentences': sentences,
        'paragraphs': paragraphs,
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'readability_score': round(calculate_readability_score(text), 2),
        'language': detect_language(text)
    }


__all__ = [
    'clean_post_content',
    'extract_keywords',
    'detect_language',
    'calculate_readability_score',
    'extract_sentences',
    'get_text_statistics'
]
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
    """Enhanced cleaning of HTML and forum-specific formatting from post content.
    
    This function removes HTML tags, forum-specific patterns, and normalizes
    whitespace while preserving important technical information and structure.
    
    Args:
        raw: Raw HTML content from forum post
        
    Returns:
        Cleaned text content optimized for vape/device discussions
    """
    if not raw:
        return ""

    try:
        soup = BeautifulSoup(raw, "html.parser")

        # Remove script and style elements completely
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Preserve important technical information before removing elements
        preserved_info = _extract_technical_info(soup)

        # Remove quote blocks but preserve quoted technical specs
        for quote in soup.select('.quote, .quotebox, blockquote'):
            quote_text = quote.get_text().strip()
            # Preserve if it contains technical specifications
            if _contains_technical_specs(quote_text):
                preserved_info.append(f"Referenced: {quote_text[:200]}")
            quote.decompose()

        # Remove signature blocks but preserve contact/shop info
        for sig in soup.select('.signature, .sig, .postbit_signature'):
            sig_text = sig.get_text().strip()
            # Preserve business/contact information
            if any(keyword in sig_text.lower() for keyword in ['shop', 'store', 'contact', 'website']):
                preserved_info.append(f"Contact: {sig_text[:100]}")
            sig.decompose()

        # Enhanced content extraction with formatting preservation
        text = _extract_structured_content(soup)

        # Add back preserved technical information
        if preserved_info:
            text += "\n\n" + "\n".join(preserved_info)

        # Clean up forum-specific patterns while preserving technical terms
        text = _clean_forum_patterns_enhanced(text)
        
        # Enhanced whitespace normalization
        text = _normalize_whitespace_enhanced(text)
        
        return text.strip()
        
    except Exception as e:
        logger.warning(f"Error cleaning post content: {e}")
        return str(raw)[:1000]  # Fallback to truncated raw content


def _extract_technical_info(soup: BeautifulSoup) -> List[str]:
    """Extract technical information before removing elements.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        List of preserved technical information
    """
    preserved = []
    
    # Extract from tables (often contain specs)
    for table in soup.find_all('table'):
        table_text = table.get_text().strip()
        if _contains_technical_specs(table_text):
            preserved.append(f"Specs: {table_text}")
    
    # Extract from code blocks (settings, commands)
    for code in soup.find_all(['code', 'pre']):
        code_text = code.get_text().strip()
        if code_text and len(code_text) < 200:
            preserved.append(f"Code: {code_text}")
    
    # Extract from emphasized technical terms
    for elem in soup.find_all(['strong', 'b', 'em', 'i']):
        text = elem.get_text().strip()
        if _is_technical_term(text):
            preserved.append(f"Important: {text}")
    
    return preserved


def _contains_technical_specs(text: str) -> bool:
    """Check if text contains technical specifications.
    
    Args:
        text: Text to check
        
    Returns:
        True if contains technical specs
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Technical indicators for vape/device forums
    technical_keywords = [
        'watt', 'volt', 'ohm', 'resistance', 'temperature', 'celsius', 'fahrenheit',
        'coil', 'battery', 'mah', 'mod', 'tank', 'atomizer', 'vg', 'pg', 'nicotine',
        'mesh', 'ceramic', 'kanthal', 'stainless', 'titanium', 'tcr', 'wattage',
        'amperage', 'voltage', 'herb', 'concentrate', 'dry', 'convection', 'conduction'
    ]
    
    # Units and measurements
    units = ['°c', '°f', 'w', 'v', 'ω', 'a', 'ml', 'g', 'mg', 'ppm']
    
    # Check for technical keywords
    keyword_count = sum(1 for keyword in technical_keywords if keyword in text_lower)
    unit_count = sum(1 for unit in units if unit in text_lower)
    
    # Check for numeric values with units
    import re
    numeric_specs = len(re.findall(r'\d+\s*(?:w|v|ohm|ω|°[cf]|ml|g|mg)', text_lower))
    
    return keyword_count >= 2 or unit_count >= 1 or numeric_specs >= 1


def _is_technical_term(text: str) -> bool:
    """Check if text is a technical term worth preserving.
    
    Args:
        text: Text to check
        
    Returns:
        True if it's a technical term
    """
    if not text or len(text) < 3 or len(text) > 50:
        return False
    
    text_lower = text.lower().strip()
    
    # Device names and brands
    brands = ['dynavap', 'storz', 'bickel', 'arizer', 'pax', 'davinci', 'vapir', 'volcano']
    models = ['mighty', 'crafty', 'solo', 'air', 'pax3', 'iq', 'ghost', 'firefly']
    
    # Technical terms
    tech_terms = [
        'convection', 'conduction', 'hybrid', 'dosing', 'capsule', 'chamber',
        'mouthpiece', 'stem', 'cooling', 'unit', 'heating', 'element'
    ]
    
    return (text_lower in brands or text_lower in models or text_lower in tech_terms or
            _contains_technical_specs(text))


def _extract_structured_content(soup: BeautifulSoup) -> str:
    """Extract content while preserving important structure.
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Structured text content
    """
    content_parts = []
    
    # Process paragraphs and divs in order
    for elem in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        text = elem.get_text().strip()
        if text and len(text) > 10:
            # Add extra spacing for headers
            if elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                content_parts.append(f"\n{text}\n")
            else:
                content_parts.append(text)
    
    # If no structured content found, fallback to all text
    if not content_parts:
        return soup.get_text()
    
    return "\n".join(content_parts)


def _clean_forum_patterns_enhanced(text: str) -> str:
    """Enhanced cleaning of forum-specific text patterns while preserving technical info.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Remove quote attribution patterns but preserve technical quotes
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip quote attribution lines unless they contain technical info
        if any(pattern in line_lower for pattern in ['said:', 'posted:', 'wrote:']):
            if not _contains_technical_specs(line):
                continue
        
        # Skip edit notices unless they mention important changes
        if line_lower.startswith('edit:') or 'last edited' in line_lower:
            if not any(keyword in line_lower for keyword in ['temperature', 'setting', 'correction', 'update']):
                continue
        
        # Preserve the line
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Remove click to expand but preserve context
    text = re.sub(r"Click to expand\.{3}", "", text)
    text = re.sub(r"Show/Hide\s+", "", text)
    
    # Remove navigation elements
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Jump to page:", "", text)
    text = re.sub(r"Quick Reply", "", text)
    
    # Clean up reaction patterns but preserve meaningful feedback
    text = re.sub(r"Like\s*\|\s*Dislike", "", text)
    text = re.sub(r"Thanks\s*\|\s*Reply(?!\s+with)", "", text)  # Keep "Reply with quote" context
    
    return text


def _normalize_whitespace_enhanced(text: str) -> str:
    """Enhanced whitespace normalization that preserves technical formatting.
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Preserve technical specifications formatting
    lines = text.split('\n')
    normalized_lines = []
    
    for line in lines:
        # Preserve indentation for technical specs or lists
        if (line.strip() and 
            (line.startswith('  ') or line.startswith('\t') or
             _contains_technical_specs(line) or
             any(char in line for char in ['•', '-', '*', '1.', '2.', '3.']))):
            # Minimal cleanup for technical content
            normalized_lines.append(re.sub(r'[\t ]+', ' ', line.rstrip()))
        else:
            # Standard cleanup for regular text
            normalized_lines.append(line.strip())
    
    text = '\n'.join(line for line in normalized_lines if line)
    
    # Convert multiple spaces to single space, but preserve technical formatting
    text = re.sub(r' {3,}', ' ', text)  # Only collapse 3+ spaces
    
    # Limit consecutive newlines but allow more for technical sections
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text


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
"""
Date parsing utilities for Forum Wisdom Miner.

This module handles parsing of various forum date formats into standardized datetime objects.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def parse_forum_date(date_str: str) -> Optional[datetime]:
    """Parse various forum date formats into datetime objects.
    
    Args:
        date_str: Raw date string from forum
        
    Returns:
        Parsed datetime object or None if parsing failed
    """
    if not date_str or date_str == 'unknown-date':
        return None
    
    # Clean the date string
    date_str = date_str.strip()
    
    # Try various date parsing strategies
    parsers = [
        _parse_iso_datetime,
        _parse_relative_time,
        _parse_us_format,
        _parse_european_format,
        _parse_timestamp,
        _parse_forum_specific_formats
    ]
    
    for parser in parsers:
        try:
            result = parser(date_str)
            if result:
                logger.debug(f"Parsed date '{date_str}' using {parser.__name__}")
                return result
        except Exception as e:
            logger.debug(f"Parser {parser.__name__} failed for '{date_str}': {e}")
            continue
    
    logger.warning(f"Could not parse date: '{date_str}'")
    return None


def _parse_iso_datetime(date_str: str) -> Optional[datetime]:
    """Parse ISO format datetime strings."""
    # ISO 8601 format: 2023-12-25T14:30:00Z or 2023-12-25T14:30:00+00:00
    iso_patterns = [
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)',
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2})',
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)',
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    ]
    
    for pattern in iso_patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                dt_str = match.group(1)
                if dt_str.endswith('Z'):
                    return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                elif '+' in dt_str or dt_str.count('-') > 2:
                    return datetime.fromisoformat(dt_str)
                else:
                    # Assume UTC if no timezone
                    return datetime.fromisoformat(dt_str).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    
    return None


def _parse_relative_time(date_str: str) -> Optional[datetime]:
    """Parse relative time expressions (e.g., '2 hours ago', 'yesterday')."""
    from datetime import timedelta
    
    now = datetime.now(timezone.utc)
    date_lower = date_str.lower()
    
    # Handle "X time ago" patterns
    relative_patterns = [
        (r'(\d+)\s*second(?:s)?\s*ago', lambda m: now - timedelta(seconds=int(m.group(1)))),
        (r'(\d+)\s*minute(?:s)?\s*ago', lambda m: now - timedelta(minutes=int(m.group(1)))),
        (r'(\d+)\s*hour(?:s)?\s*ago', lambda m: now - timedelta(hours=int(m.group(1)))),
        (r'(\d+)\s*day(?:s)?\s*ago', lambda m: now - timedelta(days=int(m.group(1)))),
        (r'(\d+)\s*week(?:s)?\s*ago', lambda m: now - timedelta(weeks=int(m.group(1)))),
        (r'(\d+)\s*month(?:s)?\s*ago', lambda m: now - timedelta(days=int(m.group(1)) * 30)),
        (r'(\d+)\s*year(?:s)?\s*ago', lambda m: now - timedelta(days=int(m.group(1)) * 365)),
    ]
    
    for pattern, calculator in relative_patterns:
        match = re.search(pattern, date_lower)
        if match:
            return calculator(match)
    
    # Handle special cases
    if 'just now' in date_lower or 'moments ago' in date_lower:
        return now
    elif 'yesterday' in date_lower:
        return now - timedelta(days=1)
    elif 'today' in date_lower:
        # Try to extract time if present
        time_match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', date_str)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            second = int(time_match.group(3)) if time_match.group(3) else 0
            return now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        return now.replace(hour=12, minute=0, second=0, microsecond=0)
    
    return None


def _parse_us_format(date_str: str) -> Optional[datetime]:
    """Parse US date formats (MM/DD/YYYY, etc.)."""
    us_patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{4})\s*(?:at\s*)?(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?',
        r'(\d{1,2})/(\d{1,2})/(\d{4})',
        r'(\d{1,2})-(\d{1,2})-(\d{4})\s*(?:at\s*)?(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?',
        r'(\d{1,2})-(\d{1,2})-(\d{4})',
    ]
    
    for pattern in us_patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                month = int(match.group(1))
                day = int(match.group(2))
                year = int(match.group(3))
                
                # Handle time if present
                if len(match.groups()) >= 6 and match.group(4):
                    hour = int(match.group(4))
                    minute = int(match.group(5))
                    second = int(match.group(6)) if match.group(6) else 0
                    
                    # Handle AM/PM
                    if len(match.groups()) >= 7 and match.group(7):
                        if match.group(7).upper() == 'PM' and hour < 12:
                            hour += 12
                        elif match.group(7).upper() == 'AM' and hour == 12:
                            hour = 0
                else:
                    hour = minute = second = 0
                
                return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
            except ValueError:
                continue
    
    return None


def _parse_european_format(date_str: str) -> Optional[datetime]:
    """Parse European date formats (DD/MM/YYYY, etc.)."""
    eu_patterns = [
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
        r'(\d{1,2})\.(\d{1,2})\.(\d{4})',  # DD.MM.YYYY
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # DD-MM-YYYY
    ]
    
    for pattern in eu_patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                day = int(match.group(1))
                month = int(match.group(2))
                year = int(match.group(3))
                
                # Only use if day > 12 (clearly not US format)
                if day > 12:
                    return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                continue
    
    return None


def _parse_timestamp(date_str: str) -> Optional[datetime]:
    """Parse Unix timestamps."""
    # Look for numeric timestamps
    timestamp_match = re.search(r'\b(\d{10,13})\b', date_str)
    if timestamp_match:
        try:
            timestamp = int(timestamp_match.group(1))
            # Handle millisecond timestamps
            if timestamp > 10**12:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (ValueError, OSError):
            pass
    
    return None


def _parse_forum_specific_formats(date_str: str) -> Optional[datetime]:
    """Parse forum-specific date formats."""
    # Common forum formats
    forum_patterns = [
        # "Dec 25, 2023 at 2:30 PM"
        r'([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})\s*(AM|PM)',
        # "December 25, 2023, 2:30 PM"
        r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4}),?\s+(\d{1,2}):(\d{2})\s*(AM|PM)',
        # "Dec 25, 2023" (without time) - COMMON FORUM FORMAT
        r'([A-Za-z]{3})\s+(\d{1,2}),\s+(\d{4})$',
        # "December 25, 2023" (without time)
        r'([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})$',
        # "25 Dec 2023"
        r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})',
        # "2023-12-25 14:30"
        r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})',
    ]
    
    month_names = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9, 'sept': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    for i, pattern in enumerate(forum_patterns):
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                if i == 0 or i == 1:  # Month name formats with time
                    month_str = match.group(1).lower()
                    month = month_names.get(month_str)
                    if not month:
                        continue
                    day = int(match.group(2))
                    year = int(match.group(3))
                    hour = int(match.group(4))
                    minute = int(match.group(5))
                    
                    if match.group(6).upper() == 'PM' and hour < 12:
                        hour += 12
                    elif match.group(6).upper() == 'AM' and hour == 12:
                        hour = 0
                    
                    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                
                elif i == 2 or i == 3:  # "Dec 25, 2023" or "December 25, 2023" (no time)
                    month_str = match.group(1).lower()
                    month = month_names.get(month_str)
                    if not month:
                        continue
                    day = int(match.group(2))
                    year = int(match.group(3))
                    return datetime(year, month, day, 12, 0, tzinfo=timezone.utc)  # Default to noon
                
                elif i == 4:  # "25 Dec 2023"
                    day = int(match.group(1))
                    month_str = match.group(2).lower()
                    month = month_names.get(month_str)
                    if not month:
                        continue
                    year = int(match.group(3))
                    return datetime(year, month, day, tzinfo=timezone.utc)
                
                elif i == 5:  # "2023-12-25 14:30"
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))
                    hour = int(match.group(4))
                    minute = int(match.group(5))
                    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
                    
            except ValueError:
                continue
    
    return None


def get_recency_score(post_date: Optional[datetime], max_age_days: int = 365) -> float:
    """Calculate a recency score for a post based on its date.
    
    Args:
        post_date: The post's datetime
        max_age_days: Maximum age in days for scoring (older posts get 0)
        
    Returns:
        Recency score between 0.0 and 1.0
    """
    if not post_date:
        return 0.1  # Small score for posts without dates
    
    now = datetime.now(timezone.utc)
    
    # Ensure post_date is timezone-aware
    if post_date.tzinfo is None:
        post_date = post_date.replace(tzinfo=timezone.utc)
    
    age_days = (now - post_date).total_seconds() / (24 * 3600)
    
    if age_days < 0:
        # Future date (likely parsing error), give moderate score
        return 0.5
    elif age_days > max_age_days:
        # Too old
        return 0.0
    else:
        # Linear decay from 1.0 to 0.0 over max_age_days
        return max(0.0, 1.0 - (age_days / max_age_days))


__all__ = ['parse_forum_date', 'get_recency_score']
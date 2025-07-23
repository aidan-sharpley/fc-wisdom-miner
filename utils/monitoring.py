"""
Monitoring utilities and decorators for Forum Wisdom Miner.

This module provides convenient decorators and context managers for
performance monitoring and analytics collection.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from utils.performance_analytics import get_performance_analyzer

logger = logging.getLogger(__name__)


@contextmanager
def monitor_operation(operation_name: str, component: Optional[str] = None,
                     metadata: Optional[Dict] = None, 
                     log_slow_threshold: float = 1.0) -> Generator[Dict, None, None]:
    """Context manager for monitoring operations.
    
    Args:
        operation_name: Name of the operation
        component: Optional component name
        metadata: Optional metadata to record
        log_slow_threshold: Log warning if operation takes longer than this
        
    Yields:
        Dictionary to add additional metadata during operation
    """
    analyzer = get_performance_analyzer()
    
    # Create full operation name
    full_name = f"{component}.{operation_name}" if component else operation_name
    
    # Start monitoring
    operation_id = analyzer.start_operation(full_name, metadata)
    start_time = time.time()
    
    # Context metadata that can be updated during operation
    context_metadata = {}
    
    try:
        yield context_metadata
        
        # Operation succeeded
        duration = time.time() - start_time
        analyzer.end_operation(operation_id, success=True, additional_metadata=context_metadata)
        
        # Log slow operations
        if duration > log_slow_threshold:
            logger.warning(f"Slow operation: {full_name} took {duration:.2f}s")
        else:
            logger.debug(f"Operation {full_name} completed in {duration:.2f}s")
            
    except Exception as e:
        # Operation failed
        duration = time.time() - start_time
        error_metadata = {
            **context_metadata,
            'error': str(e),
            'error_type': type(e).__name__
        }
        analyzer.end_operation(operation_id, success=False, additional_metadata=error_metadata)
        
        logger.error(f"Operation {full_name} failed after {duration:.2f}s: {e}")
        raise


def monitor_performance(operation_name: str = None, component: str = None,
                       log_slow_threshold: float = 1.0):
    """Decorator for monitoring function performance.
    
    Args:
        operation_name: Operation name (defaults to function name)
        component: Component name
        log_slow_threshold: Log warning if operation takes longer than this
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        nonlocal operation_name
        if operation_name is None:
            operation_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extract metadata from function arguments if available
            metadata = {}
            
            # Try to extract useful metadata from arguments
            if args and hasattr(args[0], '__class__'):
                metadata['instance_class'] = args[0].__class__.__name__
            
            if 'query' in kwargs:
                metadata['query_length'] = len(str(kwargs['query']))
            
            if 'thread_key' in kwargs:
                metadata['thread_key'] = kwargs['thread_key']
            
            with monitor_operation(operation_name, component, metadata, log_slow_threshold) as context:
                # Add result metadata after function execution
                result = func(*args, **kwargs)
                
                if isinstance(result, (list, tuple)):
                    context['result_count'] = len(result)
                elif isinstance(result, dict):
                    context['result_keys'] = len(result)
                
                return result
        
        return wrapper
    return decorator


def monitor_embedding_operation(func: Callable) -> Callable:
    """Specialized decorator for embedding operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        metadata = {}
        
        # Extract text count and length for embedding operations
        if args and isinstance(args[1], (str, list)):
            texts = args[1]
            if isinstance(texts, str):
                metadata.update({
                    'text_count': 1,
                    'total_text_length': len(texts)
                })
            else:
                metadata.update({
                    'text_count': len(texts),
                    'total_text_length': sum(len(t) for t in texts)
                })
        
        with monitor_operation(func.__name__, 'embedding', metadata) as context:
            result = func(*args, **kwargs)
            
            # Add result metadata
            if hasattr(result, '__len__'):
                context['embeddings_generated'] = len(result) if isinstance(result, list) else 1
            
            return result
    
    return wrapper


def monitor_search_operation(func: Callable) -> Callable:
    """Specialized decorator for search operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        metadata = {}
        
        # Extract search parameters
        if 'query' in kwargs or (args and len(args) > 1):
            query = kwargs.get('query') or (args[1] if len(args) > 1 else '')
            metadata['query_length'] = len(str(query))
        
        if 'top_k' in kwargs:
            metadata['requested_results'] = kwargs['top_k']
        
        with monitor_operation(func.__name__, 'search', metadata) as context:
            result = func(*args, **kwargs)
            
            # Add result metadata
            if isinstance(result, tuple) and len(result) >= 2:
                results, metadata_dict = result[:2]
                if isinstance(results, list):
                    context['results_returned'] = len(results)
                if isinstance(metadata_dict, dict):
                    context.update({
                        k: v for k, v in metadata_dict.items()
                        if k in ['search_time', 'total_candidates', 'used_hyde', 'used_reranking']
                    })
            
            return result
    
    return wrapper


def monitor_scraping_operation(func: Callable) -> Callable:
    """Specialized decorator for scraping operations."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        metadata = {}
        
        # Extract URL for scraping operations
        if args and len(args) > 1:
            url = args[1]
            if isinstance(url, str):
                metadata['url'] = url[:100]  # Truncate long URLs
        
        with monitor_operation(func.__name__, 'scraping', metadata) as context:
            result = func(*args, **kwargs)
            
            # Add result metadata for scraping
            if isinstance(result, tuple) and len(result) >= 2:
                posts, scrape_metadata = result[:2]
                if isinstance(posts, list):
                    context['posts_scraped'] = len(posts)
                if isinstance(scrape_metadata, dict):
                    context.update({
                        k: v for k, v in scrape_metadata.items()
                        if k in ['pages_scraped', 'scrape_duration']
                    })
            
            return result
    
    return wrapper


class QueryAnalyticsCollector:
    """Collects analytics specifically for query processing."""
    
    def __init__(self):
        """Initialize the query analytics collector."""
        self.analyzer = get_performance_analyzer()
        self.query_patterns = {}
        self.response_quality_scores = []
    
    def log_query(self, query: str, thread_key: str, results_count: int,
                  processing_time: float, query_analysis: Optional[Dict] = None):
        """Log a query processing event.
        
        Args:
            query: User query
            thread_key: Thread identifier  
            results_count: Number of results returned
            processing_time: Total processing time
            query_analysis: Optional query analysis results
        """
        metadata = {
            'thread_key': thread_key,
            'query_length': len(query),
            'results_count': results_count,
            'query_words': len(query.split())
        }
        
        if query_analysis:
            metadata.update({
                'is_vague': query_analysis.get('is_vague', False),
                'analytical_intent': len(query_analysis.get('analytical_intent', [])),
                'question_type': query_analysis.get('question_type', 'unknown')
            })
        
        self.analyzer.record_metric(
            'query_processing', processing_time, success=True, metadata=metadata
        )
        
        # Track query patterns
        query_type = query_analysis.get('question_type', 'unknown') if query_analysis else 'unknown'
        if query_type not in self.query_patterns:
            self.query_patterns[query_type] = 0
        self.query_patterns[query_type] += 1
    
    def get_query_analytics(self) -> Dict:
        """Get query-specific analytics.
        
        Returns:
            Dictionary with query analytics
        """
        query_metrics = self.analyzer.get_recent_metrics(hours=24, operation='query_processing')
        
        if not query_metrics:
            return {'total_queries': 0}
        
        # Calculate statistics
        total_queries = len(query_metrics)
        avg_processing_time = sum(m.duration for m in query_metrics) / total_queries
        avg_results = sum(m.metadata.get('results_count', 0) for m in query_metrics) / total_queries
        
        # Query type distribution
        query_types = {}
        for metric in query_metrics:
            q_type = metric.metadata.get('question_type', 'unknown')
            query_types[q_type] = query_types.get(q_type, 0) + 1
        
        # Vague query percentage
        vague_queries = sum(1 for m in query_metrics if m.metadata.get('is_vague', False))
        vague_percentage = vague_queries / total_queries if total_queries > 0 else 0
        
        return {
            'total_queries': total_queries,
            'avg_processing_time': avg_processing_time,
            'avg_results_count': avg_results,
            'query_type_distribution': query_types,
            'vague_query_percentage': vague_percentage,
            'query_patterns': self.query_patterns
        }


# Global query analytics collector
query_analytics = QueryAnalyticsCollector()


def get_query_analytics() -> QueryAnalyticsCollector:
    """Get the global query analytics collector."""
    return query_analytics


__all__ = [
    'monitor_operation', 'monitor_performance', 'monitor_embedding_operation',
    'monitor_search_operation', 'monitor_scraping_operation', 
    'QueryAnalyticsCollector', 'get_query_analytics'
]
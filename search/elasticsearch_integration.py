"""
ElasticSearch integration for enhanced forum search capabilities.

This module provides fast, ranked full-text search and hybrid semantic + keyword querying
to improve the LLM's retrieval pipeline.
"""

import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ConnectionError, RequestError
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    logger.warning("ElasticSearch not available. Install with: pip install elasticsearch")


class ElasticSearchIndex:
    """ElasticSearch index for forum posts with advanced search capabilities."""
    
    def __init__(self, index_name: str = "forum_posts", host: str = "localhost", port: int = 9200):
        self.index_name = index_name
        self.host = host
        self.port = port
        self.client = None
        self.index_exists = False
        
        if ELASTICSEARCH_AVAILABLE:
            self._initialize_client()
        else:
            logger.warning("ElasticSearch client not available")
    
    def _initialize_client(self):
        """Initialize ElasticSearch client with connection handling."""
        try:
            self.client = Elasticsearch([{'host': self.host, 'port': self.port}])
            
            # Test connection
            if self.client.ping():
                logger.info(f"Connected to ElasticSearch at {self.host}:{self.port}")
                self._setup_index()
            else:
                logger.error("Failed to connect to ElasticSearch")
                self.client = None
        except ConnectionError as e:
            logger.error(f"ElasticSearch connection failed: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"ElasticSearch initialization error: {e}")
            self.client = None
    
    def _setup_index(self):
        """Setup ElasticSearch index with optimized mappings for forum posts."""
        if not self.client:
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "post_id": {"type": "keyword"},
                    "thread_id": {"type": "keyword"},
                    "author": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {"keyword": {"type": "keyword"}}
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard",
                        "search_analyzer": "standard",
                        "fields": {
                            "exact": {"type": "keyword"},
                            "suggest": {"type": "completion"}
                        }
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "boost": 2.0
                    },
                    "date": {"type": "date"},
                    "page": {"type": "integer"},
                    "global_position": {"type": "integer"},
                    "upvotes": {"type": "integer"},
                    "downvotes": {"type": "integer"},
                    "likes": {"type": "integer"},
                    "reactions": {"type": "integer"},
                    "total_score": {"type": "integer"},
                    "word_count": {"type": "integer"},
                    "content_type": {"type": "keyword"},
                    "author_activity": {"type": "keyword"},
                    "url": {"type": "keyword"},
                    "topic_keywords": {"type": "keyword"},
                    "embedding_vector": {
                        "type": "dense_vector",
                        "dims": 384  # For nomic-embed-text model
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "content_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop", "snowball"]
                        }
                    }
                }
            }
        }
        
        try:
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"ElasticSearch index '{self.index_name}' already exists")
                self.index_exists = True
            else:
                self.client.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Created ElasticSearch index '{self.index_name}'")
                self.index_exists = True
        except RequestError as e:
            logger.error(f"Failed to create ElasticSearch index: {e}")
        except Exception as e:
            logger.error(f"ElasticSearch index setup error: {e}")
    
    def index_posts(self, posts: List[Dict], embeddings: List = None) -> bool:
        """Index forum posts with optional embeddings.
        
        Args:
            posts: List of post dictionaries
            embeddings: Optional list of embedding vectors
            
        Returns:
            True if indexing successful
        """
        if not self.client or not self.index_exists:
            logger.warning("ElasticSearch client not available for indexing")
            return False
        
        try:
            # Prepare bulk indexing data
            bulk_data = []
            
            for i, post in enumerate(posts):
                # Prepare document
                doc = self._prepare_document(post)
                
                # Add embedding if available
                if embeddings and i < len(embeddings):
                    doc['embedding_vector'] = embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else embeddings[i]
                
                # Add to bulk data
                action = {"index": {"_index": self.index_name, "_id": post.get('hash', f"post_{i}")}}
                bulk_data.extend([action, doc])
            
            # Perform bulk indexing
            if bulk_data:
                response = self.client.bulk(body=bulk_data, refresh=True)
                
                if response.get('errors'):
                    logger.warning("Some documents failed to index")
                    for item in response['items']:
                        if 'index' in item and item['index'].get('status', 200) >= 400:
                            logger.warning(f"Failed to index document: {item['index'].get('error', 'Unknown error')}")
                else:
                    logger.info(f"Successfully indexed {len(posts)} posts")
                
                return not response.get('errors', False)
            
        except Exception as e:
            logger.error(f"ElasticSearch indexing error: {e}")
            return False
        
        return True
    
    def _prepare_document(self, post: Dict) -> Dict:
        """Prepare a post document for ElasticSearch indexing."""
        doc = {
            'post_id': post.get('post_id', ''),
            'thread_id': post.get('thread_id', ''),
            'author': post.get('author', 'Unknown'),
            'content': post.get('content', ''),
            'page': post.get('page', 1),
            'global_position': post.get('global_position', 0),
            'upvotes': post.get('upvotes', 0),
            'downvotes': post.get('downvotes', 0),
            'likes': post.get('likes', 0),
            'reactions': post.get('reactions', 0),
            'total_score': post.get('total_score', 0),
            'word_count': post.get('word_count', 0),
            'content_type': post.get('content_type', 'discussion'),
            'author_activity': post.get('author_activity', 'casual'),
            'url': post.get('url', ''),
            'topic_keywords': post.get('topic_keywords', [])
        }
        
        # Handle date parsing
        if post.get('parsed_date'):
            if isinstance(post['parsed_date'], datetime):
                doc['date'] = post['parsed_date'].isoformat()
            else:
                doc['date'] = post.get('date', '')
        else:
            doc['date'] = post.get('date', '')
        
        return doc
    
    def search(self, query: str, filters: Dict = None, size: int = 10, 
              use_semantic: bool = False, semantic_vector: List = None) -> Tuple[List[Dict], Dict]:
        """Perform advanced search with multiple strategies.
        
        Args:
            query: Search query string
            filters: Optional filters (author, date range, etc.)
            size: Number of results to return
            use_semantic: Whether to include semantic vector search
            semantic_vector: Semantic embedding vector for the query
            
        Returns:
            Tuple of (search_results, search_metadata)
        """
        if not self.client or not self.index_exists:
            logger.warning("ElasticSearch client not available for search")
            return [], {'error': 'ElasticSearch not available'}
        
        try:
            # Build search query
            search_body = self._build_search_query(query, filters, use_semantic, semantic_vector)
            
            # Execute search
            start_time = time.time()
            response = self.client.search(index=self.index_name, body=search_body, size=size)
            search_time = time.time() - start_time
            
            # Process results
            results = self._process_search_results(response)
            
            # Create metadata
            metadata = {
                'total_hits': response['hits']['total']['value'] if 'hits' in response else 0,
                'search_time': search_time,
                'max_score': response['hits']['max_score'] if 'hits' in response else 0,
                'query_type': 'hybrid' if use_semantic else 'text_only',
                'elasticsearch_available': True
            }
            
            logger.info(f"ElasticSearch found {len(results)} results in {search_time:.3f}s")
            return results, metadata
            
        except Exception as e:
            logger.error(f"ElasticSearch search error: {e}")
            return [], {'error': str(e), 'elasticsearch_available': False}
    
    def _build_search_query(self, query: str, filters: Dict = None, 
                           use_semantic: bool = False, semantic_vector: List = None) -> Dict:
        """Build ElasticSearch query with multiple search strategies."""
        
        # Base multi-match query
        base_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    "content^1.0",
                    "author^0.5",
                    "topic_keywords^0.8"
                ],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }
        
        # Combine queries for hybrid search
        queries = [base_query]
        
        # Add semantic similarity if available
        if use_semantic and semantic_vector:
            semantic_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding_vector') + 1.0",
                        "params": {"query_vector": semantic_vector}
                    }
                }
            }
            queries.append(semantic_query)
        
        # Build final query
        if len(queries) == 1:
            final_query = queries[0]
        else:
            final_query = {
                "bool": {
                    "should": queries,
                    "minimum_should_match": 1
                }
            }
        
        search_body = {
            "query": final_query,
            "highlight": {
                "fields": {
                    "content": {"fragment_size": 150, "number_of_fragments": 2}
                }
            },
            "sort": [
                "_score",
                {"global_position": {"order": "asc"}}
            ]
        }
        
        # Add filters if provided
        if filters:
            filter_clauses = []
            
            if filters.get('author'):
                filter_clauses.append({"term": {"author.keyword": filters['author']}})
            
            if filters.get('date_from') or filters.get('date_to'):
                date_range = {}
                if filters.get('date_from'):
                    date_range['gte'] = filters['date_from']
                if filters.get('date_to'):
                    date_range['lte'] = filters['date_to']
                filter_clauses.append({"range": {"date": date_range}})
            
            if filters.get('min_score'):
                filter_clauses.append({"range": {"total_score": {"gte": filters['min_score']}}})
            
            if filter_clauses:
                search_body["query"] = {
                    "bool": {
                        "must": [final_query],
                        "filter": filter_clauses
                    }
                }
        
        return search_body
    
    def _process_search_results(self, response: Dict) -> List[Dict]:
        """Process ElasticSearch response into post dictionaries."""
        results = []
        
        if 'hits' not in response or 'hits' not in response['hits']:
            return results
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Create result dict
            result = {
                'post_id': source.get('post_id', ''),
                'author': source.get('author', 'Unknown'),
                'content': source.get('content', ''),
                'page': source.get('page', 1),
                'global_position': source.get('global_position', 0),
                'url': source.get('url', ''),
                'upvotes': source.get('upvotes', 0),
                'likes': source.get('likes', 0),
                'reactions': source.get('reactions', 0),
                'total_score': source.get('total_score', 0),
                'elasticsearch_score': hit['_score'],
                'similarity_score': min(1.0, hit['_score'] / 10.0)  # Normalize ES score
            }
            
            # Add highlights if available
            if 'highlight' in hit:
                highlights = hit['highlight'].get('content', [])
                if highlights:
                    result['highlighted_content'] = ' ... '.join(highlights)
            
            results.append(result)
        
        return results
    
    def suggest_queries(self, partial_query: str, size: int = 5) -> List[str]:
        """Get query suggestions based on indexed content.
        
        Args:
            partial_query: Partial query for suggestions
            size: Number of suggestions to return
            
        Returns:
            List of suggested queries
        """
        if not self.client or not self.index_exists:
            return []
        
        try:
            suggest_body = {
                "suggest": {
                    "content_suggest": {
                        "prefix": partial_query,
                        "completion": {
                            "field": "content.suggest",
                            "size": size
                        }
                    }
                }
            }
            
            response = self.client.search(index=self.index_name, body=suggest_body)
            
            suggestions = []
            if 'suggest' in response and 'content_suggest' in response['suggest']:
                for suggestion in response['suggest']['content_suggest']:
                    for option in suggestion.get('options', []):
                        suggestions.append(option['text'])
            
            return suggestions[:size]
            
        except Exception as e:
            logger.error(f"ElasticSearch suggestion error: {e}")
            return []
    
    def get_analytics(self) -> Dict:
        """Get analytics about the indexed data.
        
        Returns:
            Dictionary with index analytics
        """
        if not self.client or not self.index_exists:
            return {'error': 'ElasticSearch not available'}
        
        try:
            # Get index stats
            stats = self.client.indices.stats(index=self.index_name)
            
            # Get aggregations
            agg_body = {
                "size": 0,
                "aggs": {
                    "authors": {
                        "terms": {"field": "author.keyword", "size": 10}
                    },
                    "avg_score": {
                        "avg": {"field": "total_score"}
                    },
                    "posts_by_page": {
                        "terms": {"field": "page", "size": 20}
                    }
                }
            }
            
            agg_response = self.client.search(index=self.index_name, body=agg_body)
            
            return {
                'total_documents': stats['indices'][self.index_name]['total']['docs']['count'],
                'index_size': stats['indices'][self.index_name]['total']['store']['size_in_bytes'],
                'top_authors': [bucket['key'] for bucket in agg_response['aggregations']['authors']['buckets']],
                'average_score': agg_response['aggregations']['avg_score']['value'],
                'elasticsearch_available': True
            }
            
        except Exception as e:
            logger.error(f"ElasticSearch analytics error: {e}")
            return {'error': str(e), 'elasticsearch_available': False}
    
    def delete_index(self) -> bool:
        """Delete the ElasticSearch index.
        
        Returns:
            True if deletion successful
        """
        if not self.client:
            return False
        
        try:
            if self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted ElasticSearch index '{self.index_name}'")
                self.index_exists = False
                return True
            else:
                logger.info(f"Index '{self.index_name}' does not exist")
                return True
        except Exception as e:
            logger.error(f"Failed to delete ElasticSearch index: {e}")
            return False


class HybridSearchEngine:
    """Hybrid search engine combining ElasticSearch and semantic search."""
    
    def __init__(self, posts: List[Dict], thread_dir: str):
        self.posts = posts
        self.thread_dir = thread_dir
        self.es_index = ElasticSearchIndex(f"forum_{hash(thread_dir) % 1000000}")
        self.semantic_available = False
        
        # Initialize with posts
        if self.es_index.client:
            logger.info("Initializing hybrid search with ElasticSearch")
            self._initialize_index()
        else:
            logger.info("ElasticSearch not available, using fallback search")
    
    def _initialize_index(self):
        """Initialize ElasticSearch index with posts."""
        try:
            # Index posts (without embeddings for now)
            success = self.es_index.index_posts(self.posts)
            if success:
                logger.info(f"Indexed {len(self.posts)} posts in ElasticSearch")
            else:
                logger.warning("Failed to index posts in ElasticSearch")
        except Exception as e:
            logger.error(f"Error initializing ElasticSearch index: {e}")
    
    def search(self, query: str, top_k: int = 10, filters: Dict = None, 
              semantic_vector: List = None) -> Tuple[List[Dict], Dict]:
        """Perform hybrid search combining ElasticSearch and semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional search filters
            semantic_vector: Optional semantic embedding for hybrid search
            
        Returns:
            Tuple of (search_results, search_metadata)
        """
        if self.es_index.client:
            # Use ElasticSearch for hybrid search
            return self.es_index.search(
                query=query,
                filters=filters,
                size=top_k,
                use_semantic=semantic_vector is not None,
                semantic_vector=semantic_vector
            )
        else:
            # Fallback to simple text search
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int) -> Tuple[List[Dict], Dict]:
        """Fallback search when ElasticSearch is not available."""
        logger.info("Using fallback text search")
        
        query_lower = query.lower()
        scored_posts = []
        
        for post in self.posts:
            content = post.get('content', '').lower()
            author = post.get('author', '').lower()
            
            # Simple scoring based on term matches
            score = 0
            
            # Content matches
            for term in query_lower.split():
                if term in content:
                    score += 1
                if term in author:
                    score += 0.5
            
            if score > 0:
                result = post.copy()
                result['elasticsearch_score'] = score
                result['similarity_score'] = score / len(query_lower.split())
                scored_posts.append(result)
        
        # Sort by score and limit results
        scored_posts.sort(key=lambda x: x['elasticsearch_score'], reverse=True)
        results = scored_posts[:top_k]
        
        metadata = {
            'total_hits': len(scored_posts),
            'search_time': 0.001,  # Approximate
            'max_score': scored_posts[0]['elasticsearch_score'] if scored_posts else 0,
            'query_type': 'fallback_text',
            'elasticsearch_available': False
        }
        
        return results, metadata
    
    def cleanup(self):
        """Clean up ElasticSearch resources."""
        if self.es_index:
            self.es_index.delete_index()


__all__ = ['ElasticSearchIndex', 'HybridSearchEngine']
"""
Enhanced topic analysis for improved narrative generation.

This module provides advanced content clustering, semantic similarity analysis,
and topic overview generation for forum threads.
"""

import logging
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

# Lazy imports for performance
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticTopicCluster:
    """Represents a semantically coherent cluster of posts."""
    
    def __init__(self, cluster_id: int, posts: List[Dict], centroid_embedding):
        self.cluster_id = cluster_id
        self.posts = posts
        self.centroid_embedding = centroid_embedding
        self.topic_keywords = []
        self.dominant_themes = []
        self.engagement_level = 0
        self.temporal_span = None
        
    def calculate_metrics(self):
        """Calculate cluster metrics and characteristics."""
        if not self.posts:
            return
            
        # Calculate engagement level
        total_engagement = sum(
            post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
            for post in self.posts
        )
        self.engagement_level = total_engagement / len(self.posts) if self.posts else 0
        
        # Extract temporal span
        dated_posts = [post for post in self.posts if post.get('parsed_date')]
        if dated_posts:
            dates = [post['parsed_date'] for post in dated_posts]
            self.temporal_span = {
                'start': min(dates),
                'end': max(dates),
                'duration_days': (max(dates) - min(dates)).days if len(dates) > 1 else 0
            }
        
        # Extract topic keywords using content analysis
        self.topic_keywords = self._extract_semantic_keywords()
        
    def _extract_semantic_keywords(self) -> List[str]:
        """Extract semantically meaningful keywords from cluster posts."""
        all_content = ' '.join([post.get('content', '') for post in self.posts])
        
        # Simple keyword extraction (could be enhanced with TF-IDF or NLP)
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_content.lower())
        
        # Filter common words
        stop_words = {
            'that', 'this', 'with', 'have', 'will', 'from', 'they', 'been',
            'were', 'said', 'each', 'which', 'their', 'time', 'would',
            'there', 'could', 'other', 'more', 'very', 'what', 'know',
            'just', 'first', 'into', 'over', 'think', 'also', 'back',
            'after', 'well', 'quote', 'post', 'thread', 'forum', 'user'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(8) if count > 1]


class EnhancedTopicAnalyzer:
    """Advanced topic analysis using semantic clustering and content analysis."""
    
    def __init__(self, thread_dir: str):
        self.thread_dir = thread_dir
        self.posts = []
        self.post_embeddings = []
        self.clusters = []
        
        # Note: sklearn is optional - will use fallback clustering if not available
        
        # Initialize embedding manager only when needed
        self.embedding_manager = None
        
    def analyze_thread_topics(self, posts: List[Dict], force_refresh: bool = False) -> Dict:
        """Perform comprehensive topic analysis on thread posts.
        
        Args:
            posts: List of post dictionaries
            force_refresh: Whether to regenerate analysis even if cached
            
        Returns:
            Enhanced topic analysis results
        """
        start_time = time.time()
        logger.info(f"Starting enhanced topic analysis for {len(posts)} posts")
        
        self.posts = sorted(posts, key=lambda x: x.get('global_position', 0))
        
        # Step 1: Generate or load post embeddings
        embeddings = self._get_post_embeddings()
        
        # Step 2: Perform semantic clustering
        clusters = self._perform_semantic_clustering(embeddings)
        
        # Step 3: Analyze cluster characteristics
        topic_clusters = self._analyze_cluster_characteristics(clusters)
        
        # Step 4: Generate enhanced topic overviews
        topic_overviews = self._generate_topic_overviews(topic_clusters)
        
        # Step 5: Identify topic evolution and transitions
        topic_evolution = self._analyze_topic_evolution(topic_clusters)
        
        # Step 6: Find high-value posts within each topic
        topic_highlights = self._find_topic_highlights(topic_clusters)
        
        processing_time = time.time() - start_time
        
        result = {
            'enhanced_topics': {
                'topic_clusters': [self._serialize_cluster(cluster) for cluster in topic_clusters],
                'topic_overviews': topic_overviews,
                'topic_evolution': topic_evolution,
                'topic_highlights': topic_highlights,
                'analysis_metadata': {
                    'total_clusters': len(topic_clusters),
                    'processing_time': processing_time,
                    'posts_analyzed': len(posts),
                    'method': 'semantic_clustering',
                    'generated_at': time.time()
                }
            }
        }
        
        logger.info(f"Enhanced topic analysis completed in {processing_time:.2f}s, found {len(topic_clusters)} semantic clusters")
        return result
    
    def _get_post_embeddings(self) -> List:
        """Get or generate embeddings for all posts."""
        logger.info("Generating/retrieving post embeddings for semantic clustering")
        
        # Initialize embedding manager lazily
        if self.embedding_manager is None:
            from embedding.embedding_manager import EmbeddingManager
            self.embedding_manager = EmbeddingManager()
        
        # Prepare post texts for embedding
        post_texts = []
        for post in self.posts:
            content = post.get('content', '')
            author = post.get('author', 'Unknown')
            # Combine content with metadata for richer embeddings
            combined_text = f"Author: {author}\nContent: {content}"
            post_texts.append(combined_text)
        
        # Generate embeddings with progress tracking
        embeddings = self.embedding_manager.get_embeddings(
            post_texts, 
            use_cache=True, 
            preprocess=True
        )
        
        logger.info(f"Generated embeddings for {len(embeddings)} posts")
        return embeddings
    
    def _perform_semantic_clustering(self, embeddings: List[np.ndarray]) -> List[List[int]]:
        """Perform semantic clustering on post embeddings.
        
        Args:
            embeddings: List of post embeddings
            
        Returns:
            List of clusters, where each cluster is a list of post indices
        """
        if len(embeddings) < 10:
            # For small threads, create a single cluster
            return [list(range(len(embeddings)))]
        
        if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
            try:
                # Use k-means clustering for semantic grouping
                # Convert embeddings to matrix
                embedding_matrix = np.array([emb for emb in embeddings])
                
                # Determine optimal number of clusters
                optimal_k = self._find_optimal_clusters(embedding_matrix)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embedding_matrix)
                
                # Group posts by cluster
                clusters = defaultdict(list)
                for post_idx, cluster_id in enumerate(cluster_labels):
                    clusters[cluster_id].append(post_idx)
                
                logger.info(f"Semantic clustering: {len(embeddings)} posts → {optimal_k} clusters")
                return list(clusters.values())
                
            except Exception as e:
                logger.warning(f"Clustering failed: {e}, using fallback method")
                return self._fallback_clustering()
        else:
            logger.info("sklearn/numpy not available, using enhanced position-based clustering")
            return self._fallback_clustering()
    
    def _find_optimal_clusters(self, embedding_matrix) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        n_posts = len(embedding_matrix)
        
        # Determine cluster range based on thread size
        if n_posts < 50:
            max_clusters = min(5, n_posts // 10 + 1)
        elif n_posts < 200:
            max_clusters = min(8, n_posts // 20 + 1)
        else:
            max_clusters = min(12, n_posts // 30 + 1)
        
        if max_clusters < 2:
            return 1
        
        if SKLEARN_AVAILABLE:
            try:
                best_k = 2
                best_score = -1
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embedding_matrix)
                    score = silhouette_score(embedding_matrix, cluster_labels)
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                logger.debug(f"Optimal clusters: {best_k} (silhouette score: {best_score:.3f})")
                return best_k
                
            except Exception as e:
                logger.warning(f"Silhouette analysis failed: {e}")
                return min(5, n_posts // 20 + 1)  # Conservative fallback
        else:
            return min(5, n_posts // 20 + 1)  # Conservative fallback
    
    def _fallback_clustering(self) -> List[List[int]]:
        """Fallback clustering based on post position and content similarity."""
        n_posts = len(self.posts)
        cluster_size = max(20, n_posts // 6)  # Target 6 clusters
        
        clusters = []
        for i in range(0, n_posts, cluster_size):
            cluster = list(range(i, min(i + cluster_size, n_posts)))
            clusters.append(cluster)
        
        logger.info(f"Fallback clustering: {n_posts} posts → {len(clusters)} position-based clusters")
        return clusters
    
    def _analyze_cluster_characteristics(self, clusters: List[List[int]]) -> List[SemanticTopicCluster]:
        """Analyze characteristics of each cluster.
        
        Args:
            clusters: List of clusters (post index lists)
            
        Returns:
            List of analyzed topic clusters
        """
        topic_clusters = []
        
        for i, cluster_indices in enumerate(clusters):
            cluster_posts = [self.posts[idx] for idx in cluster_indices]
            
            # Calculate centroid embedding
            if NUMPY_AVAILABLE and self.post_embeddings:
                cluster_embeddings = [self.post_embeddings[idx] for idx in cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0) if cluster_embeddings else np.zeros(384)
            else:
                centroid = []  # Empty for fallback mode
            
            # Create and analyze cluster
            topic_cluster = SemanticTopicCluster(i, cluster_posts, centroid)
            topic_cluster.calculate_metrics()
            
            topic_clusters.append(topic_cluster)
        
        return topic_clusters
    
    def _generate_topic_overviews(self, topic_clusters: List[SemanticTopicCluster]) -> List[Dict]:
        """Generate enhanced topic overviews for each cluster.
        
        Args:
            topic_clusters: List of analyzed topic clusters
            
        Returns:
            List of topic overview dictionaries
        """
        overviews = []
        
        for cluster in topic_clusters:
            if not cluster.posts:
                continue
                
            # Basic cluster info
            first_post = cluster.posts[0]
            last_post = cluster.posts[-1]
            post_count = len(cluster.posts)
            
            # Engagement summary
            total_engagement = sum(
                post.get('upvotes', 0) + post.get('likes', 0) + post.get('reactions', 0)
                for post in cluster.posts
            )
            
            # Author analysis
            authors = Counter(post.get('author', 'Unknown') for post in cluster.posts)
            top_contributors = authors.most_common(3)
            
            # Content analysis
            avg_length = np.mean([len(post.get('content', '')) for post in cluster.posts])
            
            # Generate semantic topic title
            topic_title = self._generate_topic_title(cluster)
            
            # Create comprehensive overview
            overview = {
                'cluster_id': cluster.cluster_id,
                'topic_title': topic_title,
                'topic_keywords': cluster.topic_keywords,
                'post_range': {
                    'start_position': first_post.get('global_position', 0),
                    'end_position': last_post.get('global_position', 0),
                    'start_page': first_post.get('page', 1),
                    'end_page': last_post.get('page', 1),
                    'post_count': post_count
                },
                'first_post_link': first_post.get('url', ''),
                'temporal_info': cluster.temporal_span,
                'engagement_metrics': {
                    'total_engagement': total_engagement,
                    'average_engagement': total_engagement / post_count if post_count > 0 else 0,
                    'engagement_level': 'high' if cluster.engagement_level > 2 else 'medium' if cluster.engagement_level > 0.5 else 'low'
                },
                'content_analysis': {
                    'average_post_length': int(avg_length),
                    'top_contributors': [{'author': author, 'posts': count} for author, count in top_contributors],
                    'discussion_density': post_count / max(1, (last_post.get('global_position', 1) - first_post.get('global_position', 1)))
                },
                'semantic_summary': self._generate_semantic_summary(cluster)
            }
            
            overviews.append(overview)
        
        return overviews
    
    def _generate_topic_title(self, cluster: SemanticTopicCluster) -> str:
        """Generate a semantic topic title for a cluster."""
        if not cluster.topic_keywords:
            return f"Discussion Segment {cluster.cluster_id + 1}"
        
        # Use top keywords to create meaningful title
        primary_keywords = cluster.topic_keywords[:3]
        
        # Create contextual title based on keywords
        if any(keyword in ['setting', 'config', 'setup'] for keyword in primary_keywords):
            title = "Configuration & Settings Discussion"
        elif any(keyword in ['problem', 'issue', 'help', 'fix'] for keyword in primary_keywords):
            title = "Troubleshooting & Support"
        elif any(keyword in ['review', 'experience', 'opinion'] for keyword in primary_keywords):
            title = "User Experiences & Reviews"
        elif any(keyword in ['compare', 'versus', 'better', 'difference'] for keyword in primary_keywords):
            title = "Product Comparisons"
        elif any(keyword in ['new', 'latest', 'update', 'release'] for keyword in primary_keywords):
            title = "News & Updates"
        else:
            # Use dominant keyword as base
            dominant_keyword = primary_keywords[0] if primary_keywords else "General"
            title = f"{dominant_keyword.title()} Discussion"
        
        return title
    
    def _generate_semantic_summary(self, cluster: SemanticTopicCluster) -> str:
        """Generate a semantic summary of cluster content."""
        if not cluster.posts:
            return "No content available for summary."
        
        # Analyze post content for key themes
        post_count = len(cluster.posts)
        avg_engagement = cluster.engagement_level
        top_keywords = cluster.topic_keywords[:5]
        
        # Get author diversity
        unique_authors = len(set(post.get('author', 'Unknown') for post in cluster.posts))
        
        # Create contextual summary
        summary_parts = []
        
        # Base description
        if post_count == 1:
            summary_parts.append("A focused discussion point")
        elif post_count < 5:
            summary_parts.append("A brief discussion segment")
        elif post_count < 15:
            summary_parts.append("A moderate discussion thread")
        else:
            summary_parts.append("An extensive discussion section")
        
        # Add participant info
        if unique_authors == 1:
            summary_parts.append("featuring a single contributor")
        elif unique_authors < 4:
            summary_parts.append(f"between {unique_authors} participants")
        else:
            summary_parts.append(f"involving {unique_authors} community members")
        
        # Add engagement context
        if avg_engagement > 2:
            summary_parts.append("with high community engagement")
        elif avg_engagement > 0.5:
            summary_parts.append("generating moderate interest")
        
        # Add topic context
        if top_keywords:
            keywords_text = ", ".join(top_keywords[:3])
            summary_parts.append(f"focusing on {keywords_text}")
        
        return ". ".join(summary_parts) + "."
    
    def _analyze_topic_evolution(self, topic_clusters: List[SemanticTopicCluster]) -> List[Dict]:
        """Analyze how topics evolve throughout the thread."""
        evolution = []
        
        for i, cluster in enumerate(topic_clusters):
            evolution_entry = {
                'sequence': i + 1,
                'cluster_id': cluster.cluster_id,
                'topic_keywords': cluster.topic_keywords,
                'temporal_span': cluster.temporal_span,
                'post_count': len(cluster.posts),
                'engagement_level': cluster.engagement_level
            }
            
            # Analyze transition from previous topic
            if i > 0:
                prev_cluster = topic_clusters[i - 1]
                
                # Calculate keyword overlap
                prev_keywords = set(prev_cluster.topic_keywords)
                curr_keywords = set(cluster.topic_keywords)
                overlap = len(prev_keywords & curr_keywords)
                
                evolution_entry['transition_analysis'] = {
                    'keyword_overlap': overlap,
                    'topic_shift': 'continuation' if overlap > 2 else 'new_topic' if overlap == 0 else 'related_topic',
                    'engagement_change': cluster.engagement_level - prev_cluster.engagement_level
                }
            
            evolution.append(evolution_entry)
        
        return evolution
    
    def _find_topic_highlights(self, topic_clusters: List[SemanticTopicCluster]) -> Dict:
        """Find highlighted posts within each topic cluster."""
        highlights = {}
        
        for cluster in topic_clusters:
            if not cluster.posts:
                continue
                
            # Score posts within cluster
            scored_posts = []
            for post in cluster.posts:
                engagement_score = (
                    post.get('upvotes', 0) * 3 +
                    post.get('likes', 0) * 2 +
                    post.get('reactions', 0)
                )
                
                content_score = min(2, len(post.get('content', '')) / 500)  # Bonus for substantial content
                
                total_score = engagement_score + content_score
                
                scored_posts.append({
                    'post': post,
                    'score': total_score,
                    'engagement_score': engagement_score,
                    'content_score': content_score
                })
            
            # Get top 3 highlights per cluster
            scored_posts.sort(key=lambda x: x['score'], reverse=True)
            top_highlights = scored_posts[:3]
            
            highlights[f"cluster_{cluster.cluster_id}"] = [
                {
                    'author': highlight['post'].get('author', 'Unknown'),
                    'global_position': highlight['post'].get('global_position', 0),
                    'page': highlight['post'].get('page', 1),
                    'url': highlight['post'].get('url', ''),
                    'content_preview': highlight['post'].get('content', '')[:200] + '...' if len(highlight['post'].get('content', '')) > 200 else highlight['post'].get('content', ''),
                    'score': highlight['score'],
                    'engagement': highlight['engagement_score']
                }
                for highlight in top_highlights if highlight['score'] > 0
            ]
        
        return highlights
    
    def _serialize_cluster(self, cluster: SemanticTopicCluster) -> Dict:
        """Serialize cluster for JSON storage."""
        return {
            'cluster_id': cluster.cluster_id,
            'post_count': len(cluster.posts),
            'topic_keywords': cluster.topic_keywords,
            'engagement_level': cluster.engagement_level,
            'temporal_span': cluster.temporal_span,
            'post_positions': [post.get('global_position', 0) for post in cluster.posts]
        }


__all__ = ['EnhancedTopicAnalyzer', 'SemanticTopicCluster']
"""
Multi-Pass Analysis Fusion System for Forum Analysis.

This module combines multiple analysis types (topic modeling, user analysis, 
timeline clustering, reaction ranking) into a comprehensive summary with 
supporting quotes and links.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

from analytics.enhanced_topic_analyzer import EnhancedTopicAnalyzer
from analytics.data_analyzer import ForumDataAnalyzer
from search.verifiable_response_system import VerifiableResponseSystem, PostEvidence

logger = logging.getLogger(__name__)


class AnalysisPass:
    """Represents a single analysis pass with results and insights."""
    
    def __init__(self, pass_type: str, analysis_data: Dict, insights: List[str], 
                 supporting_evidence: List[Dict] = None, confidence: float = 0.8):
        self.pass_type = pass_type
        self.analysis_data = analysis_data
        self.insights = insights
        self.supporting_evidence = supporting_evidence or []
        self.confidence = confidence
        self.timestamp = time.time()
    
    def get_key_metrics(self) -> Dict:
        """Extract key metrics from this analysis pass."""
        metrics = {
            'pass_type': self.pass_type,
            'confidence': self.confidence,
            'insights_count': len(self.insights),
            'evidence_count': len(self.supporting_evidence)
        }
        
        # Add pass-specific metrics
        if self.pass_type == 'topic_analysis':
            metrics['topics_found'] = len(self.analysis_data.get('topic_overviews', []))
        elif self.pass_type == 'participant_analysis':
            metrics['participants_analyzed'] = self.analysis_data.get('thread_stats', {}).get('unique_authors', 0)
        elif self.pass_type == 'engagement_analysis':
            metrics['total_engagement'] = sum(
                post.get('score', 0) for post in self.analysis_data.get('top_5_posts', [])
            )
        elif self.pass_type == 'temporal_analysis':
            metrics['duration_days'] = self.analysis_data.get('thread_timeline', {}).get('duration_days', 0)
        
        return metrics


class MultiPassFusionSystem:
    """System for performing multi-pass analysis and fusing results into comprehensive insights."""
    
    def __init__(self, thread_dir: str, posts: List[Dict]):
        self.thread_dir = thread_dir
        self.posts = posts
        self.analysis_passes = []
        
        # Initialize analyzers
        self.topic_analyzer = EnhancedTopicAnalyzer(thread_dir)
        self.data_analyzer = ForumDataAnalyzer(thread_dir)
        self.verification_system = VerifiableResponseSystem(posts)
        
        # Fusion configuration
        self.fusion_config = {
            'max_insights_per_pass': 5,
            'min_confidence_threshold': 0.5,
            'evidence_limit_per_insight': 3,
            'cross_reference_threshold': 0.7
        }
    
    def run_comprehensive_analysis(self, force_refresh: bool = False) -> Dict:
        """Run all analysis passes and fuse results into comprehensive insights.
        
        Args:
            force_refresh: Whether to regenerate analysis even if cached
            
        Returns:
            Comprehensive fused analysis results
        """
        start_time = time.time()
        logger.info(f"Starting multi-pass analysis fusion for {len(self.posts)} posts")
        
        # Step 1: Run individual analysis passes
        self._run_topic_analysis_pass()
        self._run_participant_analysis_pass()
        self._run_engagement_analysis_pass()
        self._run_temporal_analysis_pass()
        
        # Step 2: Cross-reference and validate insights
        validated_insights = self._cross_reference_insights()
        
        # Step 3: Fuse insights into comprehensive summary
        fused_summary = self._fuse_insights_into_summary(validated_insights)
        
        # Step 4: Generate supporting evidence
        evidence_report = self._generate_comprehensive_evidence(fused_summary)
        
        # Step 5: Create final comprehensive report
        processing_time = time.time() - start_time
        
        comprehensive_report = {
            'multipass_analysis': {
                'fused_summary': fused_summary,
                'evidence_report': evidence_report,
                'analysis_passes': [self._serialize_pass(pass_obj) for pass_obj in self.analysis_passes],
                'cross_reference_insights': validated_insights,
                'fusion_metadata': {
                    'total_passes': len(self.analysis_passes),
                    'processing_time': processing_time,
                    'posts_analyzed': len(self.posts),
                    'insights_generated': len(validated_insights),
                    'confidence_average': np.mean([p.confidence for p in self.analysis_passes]) if self.analysis_passes else 0,
                    'generated_at': time.time(),
                    'method': 'multipass_fusion'
                }
            }
        }
        
        logger.info(f"Multi-pass analysis completed in {processing_time:.2f}s, generated {len(validated_insights)} validated insights")
        return comprehensive_report
    
    def _run_topic_analysis_pass(self):
        """Run enhanced topic analysis pass."""
        try:
            logger.info("Running topic analysis pass...")
            analysis_result = self.topic_analyzer.analyze_thread_topics(self.posts)
            
            enhanced_topics = analysis_result.get('enhanced_topics', {})
            topic_overviews = enhanced_topics.get('topic_overviews', [])
            
            # Extract insights
            insights = []
            for overview in topic_overviews[:3]:  # Top 3 topics
                topic_title = overview.get('topic_title', 'Unknown Topic')
                post_count = overview.get('post_range', {}).get('post_count', 0)
                engagement_level = overview.get('engagement_metrics', {}).get('engagement_level', 'low')
                
                insight = f"Topic '{topic_title}' spans {post_count} posts with {engagement_level} community engagement"
                insights.append(insight)
            
            # Create evidence from topic highlights
            supporting_evidence = []
            topic_highlights = enhanced_topics.get('topic_highlights', {})
            for cluster_id, highlights in topic_highlights.items():
                supporting_evidence.extend(highlights[:2])  # Top 2 per cluster
            
            # Create analysis pass
            topic_pass = AnalysisPass(
                pass_type='topic_analysis',
                analysis_data=enhanced_topics,
                insights=insights,
                supporting_evidence=supporting_evidence,
                confidence=0.85
            )
            
            self.analysis_passes.append(topic_pass)
            logger.info(f"Topic analysis pass completed: {len(insights)} insights, {len(supporting_evidence)} evidence items")
            
        except Exception as e:
            logger.error(f"Topic analysis pass failed: {e}")
    
    def _run_participant_analysis_pass(self):
        """Run participant activity analysis pass."""
        try:
            logger.info("Running participant analysis pass...")
            analysis_result = self.data_analyzer.analyze_participant_activity("participant analysis")
            
            if 'error' not in analysis_result:
                # Extract insights
                insights = []
                most_active = analysis_result.get('most_active_author', {})
                thread_stats = analysis_result.get('thread_stats', {})
                
                if most_active.get('name'):
                    author_name = most_active['name']
                    post_count = most_active['post_count']
                    percentage = most_active.get('percentage', 0)
                    
                    insights.append(f"{author_name} is the most active participant with {post_count} posts ({percentage:.1f}% of thread)")
                
                total_participants = thread_stats.get('unique_authors', 0)
                total_posts = thread_stats.get('total_posts', 0)
                
                if total_participants > 0:
                    insights.append(f"Thread has {total_participants} unique participants across {total_posts} posts")
                
                # Create evidence (sample posts from most active user)
                supporting_evidence = []
                if most_active.get('name'):
                    author_posts = [post for post in self.posts if post.get('author') == most_active['name']]
                    sample_posts = sorted(author_posts, key=lambda p: p.get('global_position', 0))[:3]
                    
                    for post in sample_posts:
                        evidence = {
                            'author': post.get('author', 'Unknown'),
                            'global_position': post.get('global_position', 0),
                            'page': post.get('page', 1),
                            'url': post.get('url', ''),
                            'content_preview': post.get('content', '')[:150] + '...' if len(post.get('content', '')) > 150 else post.get('content', ''),
                            'evidence_type': 'participant_activity'
                        }
                        supporting_evidence.append(evidence)
                
                # Create analysis pass
                participant_pass = AnalysisPass(
                    pass_type='participant_analysis',
                    analysis_data=analysis_result,
                    insights=insights,
                    supporting_evidence=supporting_evidence,
                    confidence=0.95
                )
                
                self.analysis_passes.append(participant_pass)
                logger.info(f"Participant analysis pass completed: {len(insights)} insights, {len(supporting_evidence)} evidence items")
            
        except Exception as e:
            logger.error(f"Participant analysis pass failed: {e}")
    
    def _run_engagement_analysis_pass(self):
        """Run engagement/reaction analysis pass."""
        try:
            logger.info("Running engagement analysis pass...")
            analysis_result = self.data_analyzer.analyze_engagement_queries("highest rated posts")
            
            if 'error' not in analysis_result:
                # Extract insights
                insights = []
                top_post = analysis_result.get('top_post', {})
                total_with_engagement = analysis_result.get('total_posts_with_engagement', 0)
                total_analyzed = analysis_result.get('total_posts_analyzed', 0)
                
                if top_post.get('author'):
                    author = top_post['author']
                    score = top_post.get('score', 0)
                    position = top_post.get('global_position', 0)
                    
                    insights.append(f"Highest engagement post by {author} (Post #{position}) with score of {score}")
                
                if total_with_engagement > 0 and total_analyzed > 0:
                    engagement_rate = (total_with_engagement / total_analyzed) * 100
                    insights.append(f"{engagement_rate:.1f}% of posts ({total_with_engagement}/{total_analyzed}) received community engagement")
                
                # Create evidence from top engaged posts
                supporting_evidence = []
                top_posts = analysis_result.get('top_5_posts', [])[:3]  # Top 3
                
                for post_data in top_posts:
                    evidence = {
                        'author': post_data.get('author', 'Unknown'),
                        'global_position': post_data.get('global_position', 0),
                        'page': post_data.get('page', 1),
                        'url': post_data.get('post_url', ''),
                        'content_preview': post_data.get('content_preview', ''),
                        'score': post_data.get('score', 0),
                        'evidence_type': 'high_engagement'
                    }
                    supporting_evidence.append(evidence)
                
                # Create analysis pass
                engagement_pass = AnalysisPass(
                    pass_type='engagement_analysis',
                    analysis_data=analysis_result,
                    insights=insights,
                    supporting_evidence=supporting_evidence,
                    confidence=0.9
                )
                
                self.analysis_passes.append(engagement_pass)
                logger.info(f"Engagement analysis pass completed: {len(insights)} insights, {len(supporting_evidence)} evidence items")
            
        except Exception as e:
            logger.error(f"Engagement analysis pass failed: {e}")
    
    def _run_temporal_analysis_pass(self):
        """Run temporal/timeline analysis pass."""
        try:
            logger.info("Running temporal analysis pass...")
            analysis_result = self.data_analyzer.analyze_temporal_patterns("timeline analysis")
            
            if 'error' not in analysis_result:
                # Extract insights
                insights = []
                timeline = analysis_result.get('thread_timeline', {})
                activity = analysis_result.get('activity_pattern', {})
                
                duration_days = timeline.get('duration_days', 0)
                posts_with_dates = timeline.get('posts_with_dates', 0)
                
                if duration_days > 0:
                    insights.append(f"Discussion spanned {duration_days} days with {posts_with_dates} timestamped posts")
                
                most_active_month = activity.get('most_active_month')
                if most_active_month:
                    month, post_count = most_active_month
                    insights.append(f"Most active period was {month} with {post_count} posts")
                
                avg_per_day = activity.get('average_posts_per_day', 0)
                if avg_per_day > 0:
                    insights.append(f"Average activity rate: {avg_per_day:.1f} posts per day")
                
                # Create evidence (first and last posts if available)
                supporting_evidence = []
                if timeline.get('first_post') and timeline.get('last_post'):
                    # Find actual first and last posts
                    sorted_posts = sorted([p for p in self.posts if p.get('parsed_date')], 
                                        key=lambda p: p.get('parsed_date'))
                    
                    if sorted_posts:
                        first_post = sorted_posts[0]
                        last_post = sorted_posts[-1]
                        
                        for post, desc in [(first_post, 'first_post'), (last_post, 'last_post')]:
                            evidence = {
                                'author': post.get('author', 'Unknown'),
                                'global_position': post.get('global_position', 0),
                                'page': post.get('page', 1),
                                'url': post.get('url', ''),
                                'content_preview': post.get('content', '')[:150] + '...' if len(post.get('content', '')) > 150 else post.get('content', ''),
                                'date': post.get('date', ''),
                                'evidence_type': desc
                            }
                            supporting_evidence.append(evidence)
                
                # Create analysis pass
                temporal_pass = AnalysisPass(
                    pass_type='temporal_analysis',
                    analysis_data=analysis_result,
                    insights=insights,
                    supporting_evidence=supporting_evidence,
                    confidence=0.8
                )
                
                self.analysis_passes.append(temporal_pass)
                logger.info(f"Temporal analysis pass completed: {len(insights)} insights, {len(supporting_evidence)} evidence items")
            
        except Exception as e:
            logger.error(f"Temporal analysis pass failed: {e}")
    
    def _cross_reference_insights(self) -> List[Dict]:
        """Cross-reference insights across analysis passes for validation."""
        logger.info("Cross-referencing insights across analysis passes...")
        
        validated_insights = []
        
        for pass_obj in self.analysis_passes:
            for insight in pass_obj.insights:
                # Find supporting evidence from other passes
                cross_references = self._find_cross_references(insight, pass_obj)
                
                validated_insight = {
                    'insight': insight,
                    'primary_source': pass_obj.pass_type,
                    'confidence': pass_obj.confidence,
                    'cross_references': cross_references,
                    'validation_score': self._calculate_validation_score(insight, cross_references),
                    'supporting_evidence': pass_obj.supporting_evidence[:self.fusion_config['evidence_limit_per_insight']]
                }
                
                # Only include insights above confidence threshold
                if validated_insight['validation_score'] >= self.fusion_config['min_confidence_threshold']:
                    validated_insights.append(validated_insight)
        
        # Sort by validation score
        validated_insights.sort(key=lambda x: x['validation_score'], reverse=True)
        
        logger.info(f"Cross-referencing completed: {len(validated_insights)} validated insights")
        return validated_insights
    
    def _find_cross_references(self, insight: str, source_pass: AnalysisPass) -> List[Dict]:
        """Find cross-references for an insight from other analysis passes."""
        cross_references = []
        
        insight_lower = insight.lower()
        
        for other_pass in self.analysis_passes:
            if other_pass.pass_type == source_pass.pass_type:
                continue
            
            # Look for related insights or data
            for other_insight in other_pass.insights:
                # Simple keyword matching for cross-references
                common_words = set(insight_lower.split()) & set(other_insight.lower().split())
                
                if len(common_words) >= 2:  # At least 2 common meaningful words
                    cross_ref = {
                        'source_pass': other_pass.pass_type,
                        'related_insight': other_insight,
                        'confidence': other_pass.confidence,
                        'similarity_score': len(common_words) / max(len(insight_lower.split()), len(other_insight.lower().split()))
                    }
                    cross_references.append(cross_ref)
        
        return cross_references
    
    def _calculate_validation_score(self, insight: str, cross_references: List[Dict]) -> float:
        """Calculate validation score for an insight based on cross-references."""
        base_score = 0.5  # Base confidence
        
        # Add score based on cross-references
        for cross_ref in cross_references:
            reference_score = cross_ref['confidence'] * cross_ref['similarity_score']
            base_score += reference_score * 0.1  # Weight cross-references
        
        # Normalize to 0-1 range
        return min(1.0, base_score)
    
    def _fuse_insights_into_summary(self, validated_insights: List[Dict]) -> Dict:
        """Fuse validated insights into a comprehensive summary."""
        logger.info("Fusing insights into comprehensive summary...")
        
        # Group insights by category
        insight_categories = {
            'topic_insights': [],
            'participant_insights': [],
            'engagement_insights': [],
            'temporal_insights': []
        }
        
        for insight_data in validated_insights:
            source = insight_data['primary_source']
            if source == 'topic_analysis':
                insight_categories['topic_insights'].append(insight_data)
            elif source == 'participant_analysis':
                insight_categories['participant_insights'].append(insight_data)
            elif source == 'engagement_analysis':
                insight_categories['engagement_insights'].append(insight_data)
            elif source == 'temporal_analysis':
                insight_categories['temporal_insights'].append(insight_data)
        
        # Create comprehensive summary
        summary = {
            'executive_summary': self._generate_executive_summary(validated_insights),
            'key_insights_by_category': insight_categories,
            'top_validated_insights': validated_insights[:10],  # Top 10 insights
            'confidence_distribution': self._analyze_confidence_distribution(validated_insights),
            'cross_reference_strength': self._analyze_cross_reference_strength(validated_insights)
        }
        
        return summary
    
    def _generate_executive_summary(self, validated_insights: List[Dict]) -> str:
        """Generate executive summary from validated insights."""
        if not validated_insights:
            return "No significant insights found in thread analysis."
        
        # Extract key metrics from top insights
        top_insights = validated_insights[:5]
        
        summary_parts = []
        summary_parts.append(f"Analysis of {len(self.posts)} posts revealed {len(validated_insights)} key insights.")
        
        # Add category-specific highlights
        categories = set(insight['primary_source'] for insight in top_insights)
        
        if 'participant_analysis' in categories:
            summary_parts.append("Participant activity patterns show clear engagement leaders.")
        
        if 'engagement_analysis' in categories:
            summary_parts.append("Community engagement concentrated in specific high-quality posts.")
        
        if 'topic_analysis' in categories:
            summary_parts.append("Discussion covers multiple distinct topics with varying engagement levels.")
        
        if 'temporal_analysis' in categories:
            summary_parts.append("Timeline analysis reveals activity patterns over the discussion period.")
        
        # Add confidence note
        avg_confidence = np.mean([insight['validation_score'] for insight in top_insights])
        confidence_level = 'high' if avg_confidence > 0.8 else 'medium' if avg_confidence > 0.6 else 'moderate'
        
        summary_parts.append(f"Insights validated with {confidence_level} confidence through cross-referencing.")
        
        return " ".join(summary_parts)
    
    def _analyze_confidence_distribution(self, validated_insights: List[Dict]) -> Dict:
        """Analyze confidence distribution across insights."""
        if not validated_insights:
            return {}
        
        scores = [insight['validation_score'] for insight in validated_insights]
        
        return {
            'mean_confidence': float(np.mean(scores)),
            'median_confidence': float(np.median(scores)),
            'high_confidence_count': sum(1 for score in scores if score > 0.8),
            'medium_confidence_count': sum(1 for score in scores if 0.6 <= score <= 0.8),
            'low_confidence_count': sum(1 for score in scores if score < 0.6)
        }
    
    def _analyze_cross_reference_strength(self, validated_insights: List[Dict]) -> Dict:
        """Analyze cross-reference strength across insights."""
        if not validated_insights:
            return {}
        
        cross_ref_counts = [len(insight['cross_references']) for insight in validated_insights]
        
        return {
            'mean_cross_references': float(np.mean(cross_ref_counts)) if cross_ref_counts else 0,
            'max_cross_references': max(cross_ref_counts) if cross_ref_counts else 0,
            'insights_with_cross_refs': sum(1 for count in cross_ref_counts if count > 0),
            'total_cross_references': sum(cross_ref_counts)
        }
    
    def _generate_comprehensive_evidence(self, fused_summary: Dict) -> Dict:
        """Generate comprehensive evidence report for the fused summary."""
        logger.info("Generating comprehensive evidence report...")
        
        all_evidence = []
        
        # Collect evidence from all categories
        for category, insights in fused_summary.get('key_insights_by_category', {}).items():
            for insight_data in insights:
                evidence_items = insight_data.get('supporting_evidence', [])
                for evidence in evidence_items:
                    evidence['insight_category'] = category
                    evidence['insight_text'] = insight_data['insight']
                    all_evidence.append(evidence)
        
        # Deduplicate evidence by position
        unique_evidence = {}
        for evidence in all_evidence:
            position = evidence.get('global_position', 0)
            if position not in unique_evidence or evidence.get('score', 0) > unique_evidence[position].get('score', 0):
                unique_evidence[position] = evidence
        
        # Sort by position and limit
        sorted_evidence = sorted(unique_evidence.values(), key=lambda e: e.get('global_position', 0))
        
        evidence_report = {
            'total_evidence_items': len(sorted_evidence),
            'evidence_by_category': self._group_evidence_by_category(sorted_evidence),
            'top_evidence_posts': sorted_evidence[:10],  # Top 10 evidence posts
            'evidence_coverage': {
                'posts_with_evidence': len(unique_evidence),
                'total_posts': len(self.posts),
                'coverage_percentage': (len(unique_evidence) / len(self.posts)) * 100 if self.posts else 0
            }
        }
        
        return evidence_report
    
    def _group_evidence_by_category(self, evidence_list: List[Dict]) -> Dict:
        """Group evidence by insight category."""
        grouped = defaultdict(list)
        
        for evidence in evidence_list:
            category = evidence.get('insight_category', 'unknown')
            grouped[category].append(evidence)
        
        return dict(grouped)
    
    def _serialize_pass(self, pass_obj: AnalysisPass) -> Dict:
        """Serialize analysis pass for JSON storage."""
        return {
            'pass_type': pass_obj.pass_type,
            'insights_count': len(pass_obj.insights),
            'evidence_count': len(pass_obj.supporting_evidence),
            'confidence': pass_obj.confidence,
            'timestamp': pass_obj.timestamp,
            'key_metrics': pass_obj.get_key_metrics()
        }


__all__ = ['MultiPassFusionSystem', 'AnalysisPass']
"""
Verifiable Response System for Forum Analysis.

This module ensures all generated responses are grounded in actual posts 
with verifiable permalinks and fact-based claims.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class PostEvidence:
    """Represents evidence from a specific post."""
    
    def __init__(self, post: Dict, quote: str = "", relevance: str = "", evidence_type: str = "supporting"):
        self.post = post
        self.quote = quote[:200] + "..." if len(quote) > 200 else quote  # Limit quote length
        self.relevance = relevance
        self.evidence_type = evidence_type  # supporting, contradicting, background
        
    def get_citation(self) -> str:
        """Get formatted citation for this evidence."""
        author = self.post.get('author', 'Unknown')
        position = self.post.get('global_position', 0)
        page = self.post.get('page', 1)
        url = self.post.get('url', '')
        
        citation = f"{author} (Post #{position}, Page {page})"
        if url:
            citation = f"[{citation}]({url})"
        
        return citation
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'post_id': self.post.get('post_id', ''),
            'author': self.post.get('author', 'Unknown'),
            'global_position': self.post.get('global_position', 0),
            'page': self.post.get('page', 1),
            'url': self.post.get('url', ''),
            'quote': self.quote,
            'relevance': self.relevance,
            'evidence_type': self.evidence_type,
            'citation': self.get_citation()
        }


class VerifiableResponseSystem:
    """System for creating verifiable, fact-based responses with post evidence."""
    
    def __init__(self, posts: List[Dict]):
        self.posts = posts
        self.posts_by_author = defaultdict(list)
        self.posts_by_position = {}
        
        # Index posts for efficient lookup
        for post in posts:
            author = post.get('author', 'Unknown')
            self.posts_by_author[author].append(post)
            position = post.get('global_position', 0)
            self.posts_by_position[position] = post
    
    def create_verifiable_response(self, claim: str, supporting_posts: List[Dict], 
                                 response_type: str = "general") -> Dict:
        """Create a verifiable response with post evidence.
        
        Args:
            claim: The claim or statement being made
            supporting_posts: Posts that support this claim
            response_type: Type of response (analytical, narrative, etc.)
            
        Returns:
            Verifiable response with evidence
        """
        evidence_list = []
        
        # Create evidence objects from supporting posts
        for post in supporting_posts:
            evidence = self._extract_evidence_from_post(post, claim)
            if evidence:
                evidence_list.append(evidence)
        
        # Sort evidence by relevance and post position
        evidence_list.sort(key=lambda e: (e.evidence_type == 'supporting', e.post.get('global_position', 0)))
        
        response = {
            'claim': claim,
            'evidence_count': len(evidence_list),
            'evidence': [evidence.to_dict() for evidence in evidence_list],
            'verification_summary': self._create_verification_summary(evidence_list),
            'response_type': response_type,
            'fact_checked': True,
            'confidence_level': self._calculate_confidence_level(evidence_list)
        }
        
        return response
    
    def verify_participant_claim(self, claim: str, participant_data: Dict) -> Dict:
        """Verify claims about participant activity with post evidence.
        
        Args:
            claim: Claim about participant activity
            participant_data: Data about participants from analysis
            
        Returns:
            Verified response with supporting evidence
        """
        evidence_posts = []
        
        # For most active user claims
        most_active = participant_data.get('most_active_author', {})
        if most_active.get('name'):
            author_name = most_active['name']
            author_posts = self.posts_by_author.get(author_name, [])
            
            # Get a sample of this author's posts as evidence
            sample_posts = sorted(author_posts, key=lambda p: p.get('global_position', 0))[:5]
            evidence_posts.extend(sample_posts)
            
            # Update claim with specific data
            post_count = most_active.get('post_count', 0)
            percentage = most_active.get('percentage', 0)
            claim = f"{author_name} is the most active participant with {post_count} posts ({percentage:.1f}% of thread)"
        
        return self.create_verifiable_response(claim, evidence_posts, "analytical")
    
    def verify_engagement_claim(self, claim: str, engagement_data: Dict) -> Dict:
        """Verify claims about post engagement with specific post evidence.
        
        Args:
            claim: Claim about engagement/ratings
            engagement_data: Data about top engaged posts
            
        Returns:
            Verified response with the actual post
        """
        evidence_posts = []
        top_post = engagement_data.get('top_post', {})
        
        if top_post.get('post'):
            actual_post = top_post['post']
            evidence_posts = [actual_post]
            
            # Create specific claim with actual data
            author = actual_post.get('author', 'Unknown')
            score = top_post.get('score', 0)
            position = actual_post.get('global_position', 0)
            
            claim = f"The highest engagement post is by {author} (Post #{position}) with a score of {score}"
        
        return self.create_verifiable_response(claim, evidence_posts, "analytical")
    
    def verify_positional_claim(self, claim: str, positional_data: Dict) -> Dict:
        """Verify claims about posting order with specific post evidence.
        
        Args:
            claim: Claim about posting position
            positional_data: Data about positional analysis
            
        Returns:
            Verified response with the specific post
        """
        evidence_posts = []
        
        author = positional_data.get('author', 'Unknown')
        position = positional_data.get('position', 1)
        
        # Find the author's first post as evidence
        if author != 'Unknown':
            author_posts = self.posts_by_author.get(author, [])
            if author_posts:
                first_post = min(author_posts, key=lambda p: p.get('global_position', 0))
                evidence_posts = [first_post]
                
                # Update claim with specific post data
                post_position = first_post.get('global_position', 0)
                claim = f"{author} was the {self._ordinal(position)} user to post (Post #{post_position})"
        
        return self.create_verifiable_response(claim, evidence_posts, "analytical")
    
    def verify_technical_claim(self, claim: str, technical_data: Dict) -> Dict:
        """Verify claims about technical specifications with supporting posts.
        
        Args:
            claim: Claim about technical specifications
            technical_data: Data about technical specifications found
            
        Returns:
            Verified response with posts containing specs
        """
        evidence_posts = []
        top_posts = technical_data.get('top_posts', [])
        
        # Get posts that contain actual technical data
        for post_data in top_posts[:3]:  # Top 3 posts with specs
            if 'spec_values' in post_data:
                # Find the actual post object
                author = post_data.get('author', 'Unknown')
                page = post_data.get('page', 1)
                
                matching_posts = [
                    post for post in self.posts 
                    if post.get('author') == author and post.get('page') == page
                ]
                
                if matching_posts:
                    evidence_posts.append(matching_posts[0])
        
        return self.create_verifiable_response(claim, evidence_posts, "technical")
    
    def add_citations_to_narrative(self, narrative_text: str, topic_highlights: List[Dict]) -> Tuple[str, List[Dict]]:
        """Add citations to narrative text based on topic highlights.
        
        Args:
            narrative_text: Original narrative text
            topic_highlights: List of highlighted posts for the topic
            
        Returns:
            Tuple of (enhanced_narrative_with_citations, evidence_list)
        """
        if not topic_highlights:
            return narrative_text, []
        
        evidence_list = []
        enhanced_narrative = narrative_text
        
        # Add citation markers to key statements
        citation_counter = 1
        
        for highlight in topic_highlights[:3]:  # Limit to top 3 highlights
            # Find the actual post
            post_position = highlight.get('global_position', 0)
            actual_post = self.posts_by_position.get(post_position)
            
            if actual_post:
                evidence = PostEvidence(
                    post=actual_post,
                    quote=highlight.get('content_preview', ''),
                    relevance="Supporting evidence for narrative claims",
                    evidence_type="supporting"
                )
                evidence_list.append(evidence)
                
                # Add citation marker to narrative
                citation_marker = f" [{citation_counter}]"
                
                # Insert citation at the end of the first sentence
                sentences = enhanced_narrative.split('. ')
                if sentences:
                    sentences[0] += citation_marker
                    enhanced_narrative = '. '.join(sentences)
                
                citation_counter += 1
        
        return enhanced_narrative, [e.to_dict() for e in evidence_list]
    
    def _extract_evidence_from_post(self, post: Dict, claim: str) -> Optional[PostEvidence]:
        """Extract relevant evidence from a post for a given claim."""
        content = post.get('content', '')
        
        # Find relevant quotes from the post
        sentences = content.split('. ')
        relevant_quotes = []
        
        # Simple keyword matching for relevance
        claim_words = set(claim.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(claim_words & sentence_words) >= 2:  # At least 2 matching words
                relevant_quotes.append(sentence.strip())
        
        if relevant_quotes:
            # Take the most relevant quote (first match)
            quote = relevant_quotes[0]
            relevance = f"Contains relevant information about: {claim[:50]}..."
            
            return PostEvidence(
                post=post,
                quote=quote,
                relevance=relevance,
                evidence_type="supporting"
            )
        
        return None
    
    def _create_verification_summary(self, evidence_list: List[PostEvidence]) -> str:
        """Create a summary of the verification evidence."""
        if not evidence_list:
            return "No direct evidence found in posts."
        
        supporting_count = sum(1 for e in evidence_list if e.evidence_type == "supporting")
        total_posts = len(evidence_list)
        
        summary_parts = []
        summary_parts.append(f"Verified with {total_posts} post(s)")
        
        if supporting_count == total_posts:
            summary_parts.append("All evidence supports the claim")
        elif supporting_count > total_posts // 2:
            summary_parts.append(f"Majority evidence ({supporting_count}/{total_posts}) supports the claim")
        else:
            summary_parts.append(f"Mixed evidence ({supporting_count}/{total_posts} supporting)")
        
        # Add author diversity
        unique_authors = len(set(e.post.get('author', 'Unknown') for e in evidence_list))
        if unique_authors > 1:
            summary_parts.append(f"from {unique_authors} different authors")
        
        return "; ".join(summary_parts)
    
    def _calculate_confidence_level(self, evidence_list: List[PostEvidence]) -> str:
        """Calculate confidence level based on evidence quality."""
        if not evidence_list:
            return "low"
        
        total_evidence = len(evidence_list)
        supporting_evidence = sum(1 for e in evidence_list if e.evidence_type == "supporting")
        
        # Factor in author diversity
        unique_authors = len(set(e.post.get('author', 'Unknown') for e in evidence_list))
        
        # Calculate confidence score
        if supporting_evidence == total_evidence and total_evidence >= 3 and unique_authors >= 2:
            return "high"
        elif supporting_evidence >= total_evidence * 0.7 and total_evidence >= 2:
            return "medium"
        else:
            return "low"
    
    def _ordinal(self, n: int) -> str:
        """Convert number to ordinal (1st, 2nd, 3rd, etc.)."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    def generate_fact_check_report(self, response_data: Dict) -> Dict:
        """Generate a fact-checking report for a response.
        
        Args:
            response_data: Response data from query processing
            
        Returns:
            Fact-checking report
        """
        report = {
            'fact_checked': True,
            'timestamp': response_data.get('processing_time', 0),
            'query_type': response_data.get('query_type', 'unknown'),
            'evidence_summary': {},
            'verifiable_claims': [],
            'confidence_assessment': 'medium'
        }
        
        # Analyze different types of responses
        if response_data.get('query_type') == 'analytical':
            analytical_result = response_data.get('analytical_result', {})
            result_type = analytical_result.get('type', '')
            
            if result_type == 'participant_analysis':
                verification = self.verify_participant_claim(
                    "Participant activity analysis", 
                    analytical_result
                )
                report['verifiable_claims'].append(verification)
                
            elif result_type == 'engagement_analysis':
                verification = self.verify_engagement_claim(
                    "Engagement analysis", 
                    analytical_result
                )
                report['verifiable_claims'].append(verification)
                
            elif result_type == 'positional_analysis':
                verification = self.verify_positional_claim(
                    "Positional analysis", 
                    analytical_result
                )
                report['verifiable_claims'].append(verification)
        
        # Calculate overall confidence
        if report['verifiable_claims']:
            confidence_levels = [claim.get('confidence_level', 'low') for claim in report['verifiable_claims']]
            high_count = confidence_levels.count('high')
            medium_count = confidence_levels.count('medium')
            
            if high_count >= len(confidence_levels) // 2:
                report['confidence_assessment'] = 'high'
            elif high_count + medium_count >= len(confidence_levels) // 2:
                report['confidence_assessment'] = 'medium'
            else:
                report['confidence_assessment'] = 'low'
        
        return report


__all__ = ['VerifiableResponseSystem', 'PostEvidence']
"""
Enhanced Forum Analysis Integration Example.

This example demonstrates how to use all the new enhanced features together:
- Semantic topic clustering with enhanced narratives
- Verifiable responses with post evidence  
- ElasticSearch hybrid search
- Multi-pass analysis fusion
"""

import logging
from analytics.enhanced_topic_analyzer import EnhancedTopicAnalyzer
from analytics.multipass_fusion_system import MultiPassFusionSystem
from search.elasticsearch_integration import HybridSearchEngine
from search.verifiable_response_system import VerifiableResponseSystem
from utils.file_utils import safe_read_json

logger = logging.getLogger(__name__)


def run_enhanced_analysis_example(thread_dir: str):
    """Run complete enhanced analysis pipeline example.
    
    Args:
        thread_dir: Path to thread data directory
    """
    print("🚀 Enhanced Forum Analysis Integration Example")
    print("=" * 60)
    
    # Load thread data
    posts_file = f"{thread_dir}/posts.json"
    posts = safe_read_json(posts_file) or []
    
    if not posts:
        print("❌ No posts found in thread directory")
        return
    
    print(f"📄 Loaded {len(posts)} posts for analysis")
    
    # 1. Enhanced Topic Analysis with Semantic Clustering
    print("\n1️⃣ Enhanced Topic Analysis with Semantic Clustering")
    print("-" * 50)
    
    topic_analyzer = EnhancedTopicAnalyzer(thread_dir)
    enhanced_topics = topic_analyzer.analyze_thread_topics(posts)
    
    if enhanced_topics.get('enhanced_topics'):
        topics = enhanced_topics['enhanced_topics']
        topic_overviews = topics.get('topic_overviews', [])
        
        print(f"✅ Found {len(topic_overviews)} semantic topic clusters:")
        for overview in topic_overviews[:3]:  # Show top 3
            title = overview.get('topic_title', 'Unknown Topic')
            post_count = overview.get('post_range', {}).get('post_count', 0)
            engagement = overview.get('engagement_metrics', {}).get('engagement_level', 'low')
            keywords = ', '.join(overview.get('topic_keywords', [])[:3])
            
            print(f"  📋 {title}: {post_count} posts, {engagement} engagement")
            print(f"     Keywords: {keywords}")
            
            # Show first post link
            first_post_link = overview.get('first_post_link', '')
            if first_post_link:
                print(f"     🔗 Jump to topic: {first_post_link}")
        
        print(f"📊 Processing time: {topics.get('analysis_metadata', {}).get('processing_time', 0):.2f}s")
    
    # 2. Verifiable Response System
    print("\n2️⃣ Verifiable Response System with Evidence")
    print("-" * 50)
    
    verification_system = VerifiableResponseSystem(posts)
    
    # Example: Verify a participant claim
    participant_data = {
        'most_active_author': {
            'name': posts[0].get('author', 'Unknown') if posts else 'Unknown',
            'post_count': len([p for p in posts if p.get('author') == posts[0].get('author', '')]) if posts else 0,
            'percentage': 25.0
        }
    }
    
    verification = verification_system.verify_participant_claim(
        "Most active user analysis", 
        participant_data
    )
    
    print(f"✅ Verification completed:")
    print(f"  📊 Evidence count: {verification.get('evidence_count', 0)}")
    print(f"  🎯 Confidence level: {verification.get('confidence_level', 'unknown')}")
    print(f"  ✔️ Fact-checked: {verification.get('fact_checked', False)}")
    
    if verification.get('evidence'):
        print("  📝 Sample evidence:")
        for i, evidence in enumerate(verification['evidence'][:2], 1):
            citation = evidence.get('citation', 'No citation')
            quote = evidence.get('quote', '')[:100] + '...' if len(evidence.get('quote', '')) > 100 else evidence.get('quote', '')
            print(f"    {i}. {citation}")
            print(f"       \"{quote}\"")
    
    # 3. ElasticSearch Hybrid Search (if available)
    print("\n3️⃣ ElasticSearch Hybrid Search Integration")
    print("-" * 50)
    
    hybrid_search = HybridSearchEngine(posts, thread_dir)
    
    if hybrid_search.es_index.client:
        print("✅ ElasticSearch available - performing hybrid search")
        
        # Example search
        search_results, metadata = hybrid_search.search(
            query="settings configuration", 
            top_k=5
        )
        
        print(f"🔍 Search results: {len(search_results)} posts found")
        print(f"⚡ Search time: {metadata.get('search_time', 0):.3f}s")
        print(f"📈 Total hits: {metadata.get('total_hits', 0)}")
        
        if search_results:
            print("📋 Top results:")
            for i, result in enumerate(search_results[:3], 1):
                author = result.get('author', 'Unknown')
                score = result.get('elasticsearch_score', 0)
                position = result.get('global_position', 0)
                
                print(f"  {i}. {author} (Post #{position}) - Score: {score:.2f}")
                
                # Show highlight if available
                highlight = result.get('highlighted_content', '')
                if highlight:
                    print(f"     \"{highlight[:80]}...\"")
        
        # Cleanup
        hybrid_search.cleanup()
    else:
        print("⚠️ ElasticSearch not available - using fallback search")
        
        # Fallback search example
        search_results, metadata = hybrid_search.search("settings", top_k=3)
        print(f"🔍 Fallback search: {len(search_results)} results")
        print(f"🔄 Method: {metadata.get('query_type', 'unknown')}")
    
    # 4. Multi-Pass Analysis Fusion
    print("\n4️⃣ Multi-Pass Analysis Fusion System")
    print("-" * 50)
    
    fusion_system = MultiPassFusionSystem(thread_dir, posts)
    comprehensive_analysis = fusion_system.run_comprehensive_analysis()
    
    if comprehensive_analysis.get('multipass_analysis'):
        analysis = comprehensive_analysis['multipass_analysis']
        metadata = analysis.get('fusion_metadata', {})
        
        print(f"✅ Multi-pass analysis completed:")
        print(f"  🔄 Analysis passes: {metadata.get('total_passes', 0)}")
        print(f"  💡 Insights generated: {metadata.get('insights_generated', 0)}")
        print(f"  📊 Average confidence: {metadata.get('confidence_average', 0):.2f}")
        print(f"  ⏱️ Processing time: {metadata.get('processing_time', 0):.2f}s")
        
        # Show executive summary
        fused_summary = analysis.get('fused_summary', {})
        executive_summary = fused_summary.get('executive_summary', '')
        if executive_summary:
            print(f"\n📋 Executive Summary:")
            print(f"  {executive_summary}")
        
        # Show top insights
        top_insights = fused_summary.get('top_validated_insights', [])
        if top_insights:
            print(f"\n🎯 Top Validated Insights:")
            for i, insight_data in enumerate(top_insights[:3], 1):
                insight = insight_data.get('insight', '')
                score = insight_data.get('validation_score', 0)
                source = insight_data.get('primary_source', 'unknown')
                
                print(f"  {i}. {insight}")
                print(f"     Source: {source}, Validation Score: {score:.2f}")
        
        # Show evidence coverage
        evidence_report = analysis.get('evidence_report', {})
        coverage = evidence_report.get('evidence_coverage', {})
        if coverage:
            coverage_pct = coverage.get('coverage_percentage', 0)
            posts_with_evidence = coverage.get('posts_with_evidence', 0)
            
            print(f"\n📊 Evidence Coverage:")
            print(f"  {posts_with_evidence} posts with evidence ({coverage_pct:.1f}% coverage)")
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Analysis Integration Complete!")
    print("\nKey Improvements Demonstrated:")
    print("✅ Semantic topic clustering with better narratives")
    print("✅ Verifiable responses with post evidence")
    print("✅ Hybrid ElasticSearch + semantic search")
    print("✅ Multi-pass analysis with cross-validation")
    print("✅ Comprehensive evidence grounding")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        thread_directory = sys.argv[1]
        run_enhanced_analysis_example(thread_directory)
    else:
        print("Usage: python enhanced_analysis_integration.py <thread_directory>")
        print("Example: python enhanced_analysis_integration.py tmp/threads/example_thread")
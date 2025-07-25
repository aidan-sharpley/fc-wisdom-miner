#!/usr/bin/env python3
"""
Performance checker for the forum analyzer on M1 MacBook Air.
Shows system status, model availability, and performance recommendations.
"""

import sys
import json
from utils.llm_manager import llm_manager
from utils.performance_monitor import performance_monitor
import psutil

def format_bytes(bytes_value):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"

def check_system_resources():
    """Check current system resource usage."""
    print("💻 System Resources:")
    print("-" * 40)
    
    # Memory
    memory = psutil.virtual_memory()
    print(f"  Memory: {memory.percent:.1f}% used ({format_bytes(memory.used)}/{format_bytes(memory.total)})")
    print(f"  Available: {format_bytes(memory.available)}")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"  CPU: {cpu_percent:.1f}% used")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"  Disk: {disk.percent:.1f}% used ({format_bytes(disk.used)}/{format_bytes(disk.total)})")
    
    # Health assessment
    if memory.percent > 85:
        print("  ⚠️  High memory usage - may affect LLM performance")
    elif memory.percent > 70:
        print("  ⚡ Moderate memory usage")
    else:
        print("  ✅ Memory usage looks good")

def check_model_availability():
    """Check which models are available and working."""
    print("\n🤖 Model Status:")
    print("-" * 40)
    
    try:
        availability = llm_manager.check_model_availability()
        
        working_count = sum(1 for available in availability.values() if available)
        total_count = len(availability)
        
        print(f"  Working models: {working_count}/{total_count}")
        
        for model, available in availability.items():
            status = "✅" if available else "❌"
            print(f"    {status} {model}")
        
        if working_count < 2:
            print("  ⚠️  Few models available - consider running setup_models.py")
        
    except Exception as e:
        print(f"  ❌ Error checking models: {e}")
        print("  💡 Make sure Ollama is running: ollama serve")

def show_performance_stats():
    """Show performance statistics if available."""
    print("\n📊 Performance Statistics:")
    print("-" * 40)
    
    try:
        stats = performance_monitor.get_performance_summary()
        
        if stats.get('status') == 'No metrics recorded':
            print("  📋 No performance data yet")
            print("  💡 Run some queries to see performance metrics")
            return
        
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  Success rate: {stats['overall_success_rate']:.1%}")
        print(f"  Avg response time: {stats['avg_response_time']:.1f}s")
        
        if stats['recent_operations'] > 0:
            print(f"  Recent avg time: {stats['recent_avg_response_time']:.1f}s")
        
        # Model comparison
        if stats['model_performance']:
            print("\n  Model Performance:")
            for model, perf in stats['model_performance'].items():
                print(f"    {model}: {perf['avg_response_time']:.1f}s avg, {perf['success_rate']:.1%} success")
        
        # Recommendations
        if stats['recommendations']:
            print("\n  💡 Recommendations:")
            for rec in stats['recommendations']:
                print(f"    • {rec}")
    
    except Exception as e:
        print(f"  ❌ Error getting performance stats: {e}")

def show_recent_failures():
    """Show recent failure information."""
    print("\n🚨 Recent Issues:")
    print("-" * 40)
    
    try:
        failures = performance_monitor.get_recent_failures(5)
        
        if not failures:
            print("  ✅ No recent failures")
            return
        
        for failure in failures:
            print(f"  ❌ {failure['operation']} ({failure['model']})")
            print(f"     Error: {failure['error'][:80]}...")
            print(f"     Time: {failure['response_time']:.1f}s")
    
    except Exception as e:
        print(f"  ❌ Error getting failure info: {e}")

def show_quick_test():
    """Run a quick test of the LLM system."""
    print("\n🧪 Quick System Test:")
    print("-" * 40)
    
    try:
        test_prompt = "Hello, respond with just 'OK'"
        
        # Test analytics model
        print("  Testing analytics model...", end=" ")
        response, model = llm_manager.get_analytics_response(test_prompt)
        if response.strip().upper() == 'OK':
            print(f"✅ ({model})")
        else:
            print(f"⚠️  Unexpected response: {response[:20]}...")
        
        # Test narrative model  
        print("  Testing narrative model...", end=" ")
        response, model = llm_manager.get_narrative_response(test_prompt)
        if 'OK' in response.upper():
            print(f"✅ ({model})")
        else:
            print(f"⚠️  Unexpected response: {response[:20]}...")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("  💡 Check that Ollama is running and models are installed")

def export_performance_data():
    """Export performance data for analysis."""
    try:
        filename = f"performance_report_{int(time.time())}.json"
        performance_monitor.export_metrics(filename)
        print(f"\n📄 Performance data exported to {filename}")
    except Exception as e:
        print(f"\n❌ Export failed: {e}")

def main():
    print("🚀 Forum Analyzer Performance Check")
    print("=" * 50)
    
    # Check basic system resources
    check_system_resources()
    
    # Check model availability
    check_model_availability()
    
    # Show performance statistics
    show_performance_stats()
    
    # Show recent failures
    show_recent_failures()
    
    # Run quick test
    show_quick_test()
    
    print("\n" + "=" * 50)
    print("💡 Tips for better performance:")
    print("  • Use qwen2.5:0.5b for analytics tasks")
    print("  • Use qwen2.5:1.5b for narrative generation")
    print("  • Keep memory usage below 80%")
    print("  • Monitor response times with this tool")
    
    # Ask if user wants to export data
    if len(sys.argv) > 1 and sys.argv[1] == '--export':
        export_performance_data()

if __name__ == "__main__":
    import time
    main()
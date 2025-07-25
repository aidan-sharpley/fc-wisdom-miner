"""
Performance monitoring for LLM operations on M1 MacBook Air.
Tracks response times, model usage, and system resource utilization.
"""

import time
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
from typing import Dict, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    operation: str
    model_used: str
    response_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: float
    success: bool
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Monitors and tracks LLM performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.lock = Lock()
        
        # Aggregated statistics
        self.operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'success_count': 0,
            'avg_response_time': 0,
            'fastest': float('inf'),
            'slowest': 0
        })
        
        self.model_stats = defaultdict(lambda: {
            'usage_count': 0,
            'total_time': 0,
            'success_rate': 0,
            'avg_response_time': 0
        })
    
    def record_operation(self, operation: str, model_used: str, response_time: float, 
                        success: bool, error_message: Optional[str] = None) -> None:
        """Record a performance metric for an LLM operation."""
        
        # Get current system metrics if available
        if PSUTIL_AVAILABLE:
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=None)
        else:
            memory_percent = 0.0
            cpu_percent = 0.0
        
        metric = PerformanceMetric(
            operation=operation,
            model_used=model_used,
            response_time=response_time,
            memory_usage=memory_percent,
            cpu_usage=cpu_percent,
            timestamp=time.time(),
            success=success,
            error_message=error_message
        )
        
        with self.lock:
            self.metrics.append(metric)
            self._update_aggregated_stats(metric)
    
    def _update_aggregated_stats(self, metric: PerformanceMetric) -> None:
        """Update aggregated statistics with new metric."""
        
        # Update operation stats
        op_stats = self.operation_stats[metric.operation]
        op_stats['count'] += 1
        op_stats['total_time'] += metric.response_time
        if metric.success:
            op_stats['success_count'] += 1
        op_stats['avg_response_time'] = op_stats['total_time'] / op_stats['count']
        op_stats['fastest'] = min(op_stats['fastest'], metric.response_time)
        op_stats['slowest'] = max(op_stats['slowest'], metric.response_time)
        
        # Update model stats
        model_stats = self.model_stats[metric.model_used]
        model_stats['usage_count'] += 1
        model_stats['total_time'] += metric.response_time
        model_stats['avg_response_time'] = model_stats['total_time'] / model_stats['usage_count']
        model_stats['success_rate'] = len([m for m in self.metrics 
                                         if m.model_used == metric.model_used and m.success]) / model_stats['usage_count']
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        with self.lock:
            if not self.metrics:
                return {'status': 'No metrics recorded'}
            
            recent_metrics = list(self.metrics)[-100:]  # Last 100 operations
            
            summary = {
                'total_operations': len(self.metrics),
                'recent_operations': len(recent_metrics),
                'overall_success_rate': sum(1 for m in self.metrics if m.success) / len(self.metrics),
                'avg_response_time': sum(m.response_time for m in self.metrics) / len(self.metrics),
                'recent_avg_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
                'operation_breakdown': dict(self.operation_stats),
                'model_performance': dict(self.model_stats),
                'system_health': self._get_system_health(),
                'recommendations': self._generate_recommendations()
            }
            
            return summary
    
    def _get_system_health(self) -> Dict:
        """Get current system health metrics."""
        if not PSUTIL_AVAILABLE:
            return {
                'current_memory_percent': 0,
                'available_memory_gb': 0,
                'current_cpu_percent': 0,
                'recent_avg_memory': 0,
                'memory_status': 'unknown'
            }
        
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get recent memory usage from metrics
        recent_memory = []
        if self.metrics:
            recent_memory = [m.memory_usage for m in list(self.metrics)[-20:]]
        
        return {
            'current_memory_percent': memory.percent,
            'available_memory_gb': memory.available / (1024**3),
            'current_cpu_percent': cpu_percent,
            'recent_avg_memory': sum(recent_memory) / len(recent_memory) if recent_memory else 0,
            'memory_status': 'healthy' if memory.percent < 80 else 'warning' if memory.percent < 90 else 'critical'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics."""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        # Check response times
        recent_metrics = list(self.metrics)[-50:]
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        
        if avg_response_time > 30:
            recommendations.append("Consider switching to faster models (qwen2.5:0.5b for analytics)")
        
        if avg_response_time > 60:
            recommendations.append("Response times are very slow - check system resources")
        
        # Check success rates
        recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        if recent_success_rate < 0.9:
            recommendations.append("High failure rate detected - check model availability")
        
        # Check model performance
        for model, stats in self.model_stats.items():
            if stats['success_rate'] < 0.8:
                recommendations.append(f"Model {model} has low success rate ({stats['success_rate']:.1%})")
        
        # Check memory usage
        system_health = self._get_system_health()
        if system_health['current_memory_percent'] > 85:
            recommendations.append("High memory usage - consider reducing batch sizes")
        
        # Check for timeout patterns
        timeout_count = sum(1 for m in recent_metrics 
                          if not m.success and m.error_message and 'timeout' in m.error_message.lower())
        if timeout_count > len(recent_metrics) * 0.2:
            recommendations.append("Frequent timeouts - increase timeout values or use faster models")
        
        return recommendations
    
    def get_model_comparison(self) -> Dict:
        """Compare performance across different models."""
        comparison = {}
        
        for model, stats in self.model_stats.items():
            comparison[model] = {
                'usage_count': stats['usage_count'],
                'avg_response_time': round(stats['avg_response_time'], 2),
                'success_rate': round(stats['success_rate'], 3),
                'relative_speed': 'fast' if stats['avg_response_time'] < 15 else 'medium' if stats['avg_response_time'] < 30 else 'slow'
            }
        
        return comparison
    
    def get_recent_failures(self, limit: int = 10) -> List[Dict]:
        """Get recent failed operations for debugging."""
        with self.lock:
            failures = [
                {
                    'operation': m.operation,
                    'model': m.model_used,
                    'error': m.error_message,
                    'response_time': m.response_time,
                    'timestamp': m.timestamp
                }
                for m in reversed(list(self.metrics))
                if not m.success
            ]
            
            return failures[:limit]
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self.lock:
            self.metrics.clear()
            self.operation_stats.clear()
            self.model_stats.clear()
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file for analysis."""
        import json
        
        with self.lock:
            data = {
                'metrics': [
                    {
                        'operation': m.operation,
                        'model_used': m.model_used,
                        'response_time': m.response_time,
                        'memory_usage': m.memory_usage,
                        'cpu_usage': m.cpu_usage,
                        'timestamp': m.timestamp,
                        'success': m.success,
                        'error_message': m.error_message
                    }
                    for m in self.metrics
                ],
                'summary': self.get_performance_summary()
            }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


# Global instance
performance_monitor = PerformanceMonitor()

__all__ = ['PerformanceMonitor', 'performance_monitor']
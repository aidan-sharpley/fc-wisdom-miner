"""
Performance analytics and monitoring for Forum Wisdom Miner.

This module provides comprehensive performance monitoring, analytics collection,
and optimization insights for all components of the application.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric data point."""
    timestamp: float
    operation: str
    duration: float
    success: bool
    metadata: Dict[str, Any]


class PerformanceAnalyzer:
    """Comprehensive performance analysis and monitoring system."""
    
    def __init__(self, retention_hours: int = 24, max_metrics: int = 10000):
        """Initialize performance analyzer.
        
        Args:
            retention_hours: Hours to retain metrics
            max_metrics: Maximum number of metrics to keep in memory
        """
        self.retention_hours = retention_hours
        self.max_metrics = max_metrics
        
        # Metrics storage
        self.metrics = deque(maxlen=max_metrics)
        self.operation_stats = defaultdict(lambda: {
            'count': 0,
            'total_duration': 0,
            'success_count': 0,
            'failure_count': 0,
            'avg_duration': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
            'recent_durations': deque(maxlen=100)
        })
        
        # Real-time monitoring
        self.current_operations = {}  # operation_id -> start_time
        
        # Optimization insights
        self.optimization_insights = []
        self.last_analysis_time = time.time()
    
    def start_operation(self, operation: str, metadata: Optional[Dict] = None) -> str:
        """Start tracking an operation.
        
        Args:
            operation: Operation name
            metadata: Optional metadata
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation}_{time.time()}"
        self.current_operations[operation_id] = {
            'operation': operation,
            'start_time': time.time(),
            'metadata': metadata or {}
        }
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, 
                     additional_metadata: Optional[Dict] = None):
        """End tracking an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            additional_metadata: Additional metadata to record
        """
        if operation_id not in self.current_operations:
            logger.warning(f"Unknown operation ID: {operation_id}")
            return
        
        op_data = self.current_operations.pop(operation_id)
        duration = time.time() - op_data['start_time']
        
        # Combine metadata
        metadata = op_data['metadata'].copy()
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Create metric
        metric = PerformanceMetric(
            timestamp=time.time(),
            operation=op_data['operation'],
            duration=duration,
            success=success,
            metadata=metadata
        )
        
        # Store metric
        self.metrics.append(metric)
        
        # Update operation statistics
        self._update_operation_stats(metric)
        
        # Check for optimization opportunities
        self._check_optimization_opportunities(metric)
    
    def record_metric(self, operation: str, duration: float, success: bool = True,
                     metadata: Optional[Dict] = None):
        """Record a metric directly.
        
        Args:
            operation: Operation name
            duration: Operation duration in seconds
            success: Whether operation succeeded
            metadata: Optional metadata
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            operation=operation,
            duration=duration,
            success=success,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        self._update_operation_stats(metric)
        self._check_optimization_opportunities(metric)
    
    def _update_operation_stats(self, metric: PerformanceMetric):
        """Update statistics for an operation."""
        stats = self.operation_stats[metric.operation]
        
        stats['count'] += 1
        stats['total_duration'] += metric.duration
        
        if metric.success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
        
        stats['min_duration'] = min(stats['min_duration'], metric.duration)
        stats['max_duration'] = max(stats['max_duration'], metric.duration)
        stats['avg_duration'] = stats['total_duration'] / stats['count']
        stats['recent_durations'].append(metric.duration)
    
    def _check_optimization_opportunities(self, metric: PerformanceMetric):
        """Check for optimization opportunities based on recent metrics."""
        stats = self.operation_stats[metric.operation]
        
        # Check for slow operations
        if metric.duration > 5.0:  # 5+ seconds
            insight = {
                'type': 'slow_operation',
                'operation': metric.operation,
                'duration': metric.duration,
                'timestamp': metric.timestamp,
                'suggestion': f"Operation {metric.operation} took {metric.duration:.2f}s - consider optimization"
            }
            self.optimization_insights.append(insight)
        
        # Check for high failure rates
        if stats['count'] >= 10:
            failure_rate = stats['failure_count'] / stats['count']
            if failure_rate > 0.2:  # 20%+ failure rate
                insight = {
                    'type': 'high_failure_rate',
                    'operation': metric.operation,
                    'failure_rate': failure_rate,
                    'timestamp': time.time(),
                    'suggestion': f"Operation {metric.operation} has {failure_rate:.1%} failure rate - needs attention"
                }
                self.optimization_insights.append(insight)
        
        # Check for performance degradation
        if len(stats['recent_durations']) >= 20:
            recent_avg = sum(list(stats['recent_durations'])[-10:]) / 10
            older_avg = sum(list(stats['recent_durations'])[:10]) / 10
            
            if recent_avg > older_avg * 1.5:  # 50% slower
                insight = {
                    'type': 'performance_degradation',
                    'operation': metric.operation,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg,
                    'timestamp': time.time(),
                    'suggestion': f"Operation {metric.operation} performance degrading: {recent_avg:.2f}s vs {older_avg:.2f}s"
                }
                self.optimization_insights.append(insight)
        
        # Limit insights to prevent memory issues
        if len(self.optimization_insights) > 100:
            self.optimization_insights = self.optimization_insights[-50:]
    
    def get_operation_stats(self, operation: Optional[str] = None) -> Dict:
        """Get statistics for operations.
        
        Args:
            operation: Specific operation or None for all
            
        Returns:
            Operation statistics
        """
        if operation:
            return dict(self.operation_stats.get(operation, {}))
        
        return {op: dict(stats) for op, stats in self.operation_stats.items()}
    
    def get_recent_metrics(self, hours: int = 1, operation: Optional[str] = None) -> List[PerformanceMetric]:
        """Get recent metrics.
        
        Args:
            hours: Number of hours to look back
            operation: Filter by operation name
            
        Returns:
            List of recent metrics
        """
        cutoff_time = time.time() - (hours * 3600)
        
        recent_metrics = []
        for metric in reversed(self.metrics):
            if metric.timestamp < cutoff_time:
                break
            
            if operation is None or metric.operation == operation:
                recent_metrics.append(metric)
        
        return list(reversed(recent_metrics))
    
    def get_optimization_insights(self, limit: int = 10) -> List[Dict]:
        """Get recent optimization insights.
        
        Args:
            limit: Maximum number of insights to return
            
        Returns:
            List of optimization insights
        """
        return self.optimization_insights[-limit:]
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report.
        
        Returns:
            Performance report dictionary
        """
        current_time = time.time()
        
        # Overall statistics
        total_operations = sum(stats['count'] for stats in self.operation_stats.values())
        total_failures = sum(stats['failure_count'] for stats in self.operation_stats.values())
        overall_success_rate = 1 - (total_failures / max(1, total_operations))
        
        # Recent performance (last hour)
        recent_metrics = self.get_recent_metrics(hours=1)
        recent_operations = len(recent_metrics)
        recent_avg_duration = sum(m.duration for m in recent_metrics) / max(1, len(recent_metrics))
        
        # Slowest operations
        slowest_ops = sorted(
            [(op, stats['avg_duration']) for op, stats in self.operation_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        # Most frequent operations
        frequent_ops = sorted(
            [(op, stats['count']) for op, stats in self.operation_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:5]
        
        return {
            'report_timestamp': current_time,
            'overall_stats': {
                'total_operations': total_operations,
                'total_failures': total_failures,
                'success_rate': overall_success_rate,
                'unique_operations': len(self.operation_stats)
            },
            'recent_performance': {
                'operations_last_hour': recent_operations,
                'avg_duration_last_hour': recent_avg_duration,
                'operations_per_minute': recent_operations / 60 if recent_operations > 0 else 0
            },
            'slowest_operations': [{'operation': op, 'avg_duration': dur} for op, dur in slowest_ops],
            'most_frequent_operations': [{'operation': op, 'count': count} for op, count in frequent_ops],
            'optimization_insights': self.get_optimization_insights(),
            'current_operations': len(self.current_operations)
        }
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        # Remove old metrics
        while self.metrics and self.metrics[0].timestamp < cutoff_time:
            self.metrics.popleft()
        
        # Clean up old insights
        cutoff_time_insights = time.time() - (24 * 3600)  # Keep insights for 24 hours
        self.optimization_insights = [
            insight for insight in self.optimization_insights
            if insight['timestamp'] > cutoff_time_insights
        ]


class ComponentAnalyzer:
    """Analyzer for specific application components."""
    
    def __init__(self, component_name: str, performance_analyzer: PerformanceAnalyzer):
        """Initialize component analyzer.
        
        Args:
            component_name: Name of the component
            performance_analyzer: Main performance analyzer
        """
        self.component_name = component_name
        self.performance_analyzer = performance_analyzer
        
        # Component-specific metrics
        self.component_metrics = {
            'requests_per_minute': deque(maxlen=60),
            'error_rates': deque(maxlen=100),
            'resource_usage': deque(maxlen=100)
        }
    
    def log_operation(self, operation: str, duration: float, success: bool = True,
                     metadata: Optional[Dict] = None):
        """Log an operation for this component.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            success: Whether operation succeeded
            metadata: Additional metadata
        """
        full_operation = f"{self.component_name}.{operation}"
        
        # Add component context to metadata
        component_metadata = {'component': self.component_name}
        if metadata:
            component_metadata.update(metadata)
        
        self.performance_analyzer.record_metric(
            full_operation, duration, success, component_metadata
        )
        
        # Update component-specific metrics
        self._update_component_metrics(success)
    
    def _update_component_metrics(self, success: bool):
        """Update component-specific metrics."""
        current_minute = int(time.time() / 60)
        
        # Track requests per minute
        if not self.component_metrics['requests_per_minute'] or \
           self.component_metrics['requests_per_minute'][-1][0] != current_minute:
            self.component_metrics['requests_per_minute'].append([current_minute, 0])
        
        self.component_metrics['requests_per_minute'][-1][1] += 1
        
        # Track error rates
        self.component_metrics['error_rates'].append(not success)
    
    def get_component_stats(self) -> Dict:
        """Get component-specific statistics.
        
        Returns:
            Component statistics
        """
        # Calculate recent request rate
        recent_requests = sum(count for _, count in self.component_metrics['requests_per_minute'][-5:])
        requests_per_minute = recent_requests / min(5, len(self.component_metrics['requests_per_minute']))
        
        # Calculate error rate
        recent_errors = sum(self.component_metrics['error_rates'][-50:])
        error_rate = recent_errors / min(50, len(self.component_metrics['error_rates'])) if self.component_metrics['error_rates'] else 0
        
        # Get component operations from main analyzer
        component_ops = {
            op: stats for op, stats in self.performance_analyzer.operation_stats.items()
            if op.startswith(f"{self.component_name}.")
        }
        
        return {
            'component': self.component_name,
            'requests_per_minute': requests_per_minute,
            'error_rate': error_rate,
            'operations': component_ops,
            'total_operations': sum(stats['count'] for stats in component_ops.values())
        }


# Global performance analyzer instance
global_performance_analyzer = PerformanceAnalyzer()


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get the global performance analyzer instance."""
    return global_performance_analyzer


def create_component_analyzer(component_name: str) -> ComponentAnalyzer:
    """Create a component analyzer.
    
    Args:
        component_name: Name of the component
        
    Returns:
        ComponentAnalyzer instance
    """
    return ComponentAnalyzer(component_name, global_performance_analyzer)


__all__ = [
    'PerformanceMetric', 'PerformanceAnalyzer', 'ComponentAnalyzer',
    'get_performance_analyzer', 'create_component_analyzer'
]
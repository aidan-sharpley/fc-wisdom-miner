"""
Enhanced processing pipeline for Forum Wisdom Miner.

This module provides robust pipeline processing with error recovery,
progress tracking, and performance optimization.
"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Robust processing pipeline with error recovery and progress tracking."""
    
    def __init__(self, name: str = "Pipeline"):
        """Initialize the processing pipeline.
        
        Args:
            name: Pipeline name for logging
        """
        self.name = name
        self.stages = []
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_duration': 0,
            'total_duration': 0,
            'stage_stats': {}
        }
        
    def add_stage(self, name: str, function: Callable, 
                 retry_count: int = 3, critical: bool = True, 
                 recovery_function: Optional[Callable] = None):
        """Add a processing stage to the pipeline.
        
        Args:
            name: Stage name
            function: Processing function
            retry_count: Number of retries on failure
            critical: Whether failure should stop the pipeline
            recovery_function: Optional recovery function for failures
        """
        stage = {
            'name': name,
            'function': function,
            'retry_count': retry_count,
            'critical': critical,
            'recovery_function': recovery_function,
            'stats': {
                'executions': 0,
                'successes': 0,
                'failures': 0,
                'retries': 0,
                'total_duration': 0,
                'average_duration': 0
            }
        }
        self.stages.append(stage)
        self.stats['stage_stats'][name] = stage['stats']
        
    def execute(self, data: Any, progress_callback: Optional[Callable] = None) -> Tuple[Any, Dict]:
        """Execute the pipeline on input data.
        
        Args:
            data: Input data
            progress_callback: Optional progress callback function
            
        Returns:
            Tuple of (processed_data, execution_stats)
        """
        start_time = time.time()
        self.stats['total_runs'] += 1
        
        logger.info(f"Starting pipeline execution: {self.name}")
        
        current_data = data
        stage_results = {}
        failed_stages = []
        
        try:
            for i, stage in enumerate(self.stages):
                stage_name = stage['name']
                
                # Progress callback
                if progress_callback:
                    progress = (i / len(self.stages)) * 100
                    progress_callback(progress, f"Processing: {stage_name}")
                
                logger.info(f"Executing stage: {stage_name}")
                
                # Execute stage with retry logic
                try:
                    stage_result = self._execute_stage(stage, current_data)
                    current_data = stage_result
                    stage_results[stage_name] = {'success': True, 'data': stage_result}
                    
                except Exception as e:
                    logger.error(f"Stage {stage_name} failed: {e}")
                    failed_stages.append(stage_name)
                    stage_results[stage_name] = {'success': False, 'error': str(e)}
                    
                    # Try recovery if available
                    if stage['recovery_function']:
                        try:
                            logger.info(f"Attempting recovery for stage: {stage_name}")
                            recovery_result = stage['recovery_function'](current_data, e)
                            current_data = recovery_result
                            stage_results[stage_name]['recovery'] = True
                            logger.info(f"Recovery successful for stage: {stage_name}")
                        except Exception as recovery_error:
                            logger.error(f"Recovery failed for stage {stage_name}: {recovery_error}")
                            stage_results[stage_name]['recovery_error'] = str(recovery_error)
                    
                    # Stop pipeline if stage is critical
                    if stage['critical']:
                        logger.error(f"Critical stage {stage_name} failed, stopping pipeline")
                        break
            
            # Final progress callback
            if progress_callback:
                progress_callback(100, "Pipeline completed")
            
            # Update statistics
            duration = time.time() - start_time
            self.stats['total_duration'] += duration
            
            if not failed_stages or not any(self.stages[self._get_stage_index(name)]['critical'] 
                                          for name in failed_stages):
                self.stats['successful_runs'] += 1
            else:
                self.stats['failed_runs'] += 1
                
            self.stats['average_duration'] = self.stats['total_duration'] / self.stats['total_runs']
            
            execution_stats = {
                'duration': duration,
                'stages_executed': len(stage_results),
                'failed_stages': failed_stages,
                'stage_results': stage_results,
                'success': len(failed_stages) == 0 or not any(
                    self.stages[self._get_stage_index(name)]['critical'] for name in failed_stages
                )
            }
            
            logger.info(f"Pipeline execution completed in {duration:.2f}s")
            return current_data, execution_stats
            
        except Exception as e:
            self.stats['failed_runs'] += 1
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _execute_stage(self, stage: Dict, data: Any) -> Any:
        """Execute a single stage with retry logic.
        
        Args:
            stage: Stage configuration
            data: Input data
            
        Returns:
            Processed data
        """
        stage_stats = stage['stats']
        stage_stats['executions'] += 1
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(stage['retry_count'] + 1):
            try:
                result = stage['function'](data)
                
                # Update statistics
                duration = time.time() - start_time
                stage_stats['successes'] += 1
                stage_stats['total_duration'] += duration
                stage_stats['average_duration'] = (
                    stage_stats['total_duration'] / stage_stats['successes']
                )
                
                if attempt > 0:
                    stage_stats['retries'] += attempt
                    logger.info(f"Stage {stage['name']} succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < stage['retry_count']:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Stage {stage['name']} failed (attempt {attempt + 1}), "
                                 f"retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Stage {stage['name']} failed after {attempt + 1} attempts: {e}")
        
        # All retries failed
        stage_stats['failures'] += 1
        raise last_error
    
    def _get_stage_index(self, stage_name: str) -> int:
        """Get index of stage by name."""
        for i, stage in enumerate(self.stages):
            if stage['name'] == stage_name:
                return i
        return -1
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        return {
            'pipeline_name': self.name,
            'overall_stats': {
                k: v for k, v in self.stats.items() if k != 'stage_stats'
            },
            'stage_stats': self.stats['stage_stats']
        }
    
    def reset_stats(self):
        """Reset all statistics."""
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'average_duration': 0,
            'total_duration': 0,
            'stage_stats': {}
        }
        
        for stage in self.stages:
            stage['stats'] = {
                'executions': 0,
                'successes': 0,  
                'failures': 0,
                'retries': 0,
                'total_duration': 0,
                'average_duration': 0
            }
            self.stats['stage_stats'][stage['name']] = stage['stats']


@contextmanager
def performance_monitor(operation_name: str, 
                       threshold_seconds: float = 1.0) -> Generator[Dict, None, None]:
    """Context manager for monitoring operation performance.
    
    Args:
        operation_name: Name of the operation being monitored
        threshold_seconds: Log warning if operation takes longer than this
        
    Yields:
        Performance metrics dictionary
    """
    start_time = time.time()
    metrics = {
        'operation': operation_name,
        'start_time': start_time,
        'duration': 0,
        'memory_usage': 0
    }
    
    try:
        # Get initial memory usage if available
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            metrics['initial_memory'] = initial_memory
        except ImportError:
            initial_memory = 0
        
        yield metrics
        
    finally:
        end_time = time.time()
        duration = end_time - start_time
        metrics['duration'] = duration
        
        # Get final memory usage if available
        try:
            if 'psutil' in locals():
                final_memory = process.memory_info().rss
                metrics['final_memory'] = final_memory
                metrics['memory_usage'] = final_memory - initial_memory
        except:
            pass
        
        # Log performance
        if duration > threshold_seconds:
            logger.warning(f"Slow operation: {operation_name} took {duration:.2f}s")
        else:
            logger.debug(f"Operation: {operation_name} completed in {duration:.2f}s")


class BatchProcessor:
    """Efficient batch processing with progress tracking and error handling."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        """Initialize batch processor.
        
        Args:
            batch_size: Size of each processing batch
            max_workers: Maximum number of worker threads
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.stats = {
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'batches_processed': 0,
            'processing_time': 0
        }
    
    def process_items(self, items: List[Any], process_function: Callable,
                     progress_callback: Optional[Callable] = None,
                     error_callback: Optional[Callable] = None) -> Tuple[List[Any], List[Any]]:
        """Process items in batches with error handling.
        
        Args:
            items: Items to process
            process_function: Function to apply to each item
            progress_callback: Optional progress callback
            error_callback: Optional error callback
            
        Returns:
            Tuple of (successful_results, failed_items)
        """
        start_time = time.time()
        
        self.stats['total_items'] = len(items)
        successful_results = []
        failed_items = []
        
        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_start = time.time()
            
            logger.debug(f"Processing batch {i//self.batch_size + 1} "
                        f"of {(len(items) + self.batch_size - 1) // self.batch_size}")
            
            # Process batch items
            for item in batch:
                try:
                    result = process_function(item)
                    successful_results.append(result)
                    self.stats['processed_items'] += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process item: {e}")
                    failed_items.append({'item': item, 'error': str(e)})
                    self.stats['failed_items'] += 1
                    
                    if error_callback:
                        error_callback(item, e)
            
            self.stats['batches_processed'] += 1
            batch_duration = time.time() - batch_start
            
            # Progress callback
            if progress_callback:
                progress = (i + len(batch)) / len(items) * 100
                progress_callback(progress, f"Processed {i + len(batch)}/{len(items)} items")
        
        self.stats['processing_time'] = time.time() - start_time
        
        logger.info(f"Batch processing completed: {self.stats['processed_items']} successful, "
                   f"{self.stats['failed_items']} failed in {self.stats['processing_time']:.2f}s")
        
        return successful_results, failed_items
    
    def get_stats(self) -> Dict:
        """Get batch processing statistics."""
        success_rate = (self.stats['processed_items'] / 
                       max(1, self.stats['total_items']))
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'items_per_second': (self.stats['processed_items'] / 
                               max(0.1, self.stats['processing_time']))
        }


__all__ = ['ProcessingPipeline', 'performance_monitor', 'BatchProcessor']
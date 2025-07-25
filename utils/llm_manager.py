"""
Multi-model LLM manager with progressive fallback strategy.
Optimized for M1 MacBook Air with 8GB RAM.
"""

import logging
import time
from typing import Dict, Optional, List, Tuple
from enum import Enum
import requests

from config.settings import (
    OLLAMA_BASE_URL, OLLAMA_ANALYTICS_MODEL, OLLAMA_NARRATIVE_MODEL, 
    OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FAST, LLM_TIMEOUT_NARRATIVE, 
    LLM_TIMEOUT_FALLBACK, OLLAMA_CHAT_MODEL
)
# Lazy import to avoid circular dependencies
performance_monitor = None

logger = logging.getLogger(__name__)


class TaskType(Enum):
    ANALYTICS = "analytics"
    NARRATIVE = "narrative" 
    STRUCTURED = "structured"
    CREATIVE = "creative"


class ModelConfig:
    def __init__(self, name: str, timeout: int, temperature: float = 0.3, 
                 top_p: float = 0.8, max_tokens: int = 500):
        self.name = name
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens


class LLMManager:
    """Manages multiple LLM models with progressive fallback."""
    
    def __init__(self):
        self.chat_url = f"{OLLAMA_BASE_URL}/api/chat"
        self.models = self._initialize_models()
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'fallbacks_used': 0,
            'model_usage': {},
            'avg_response_time': {}
        }
        self._performance_monitor = None
    
    def _initialize_models(self) -> Dict[TaskType, List[ModelConfig]]:
        """Initialize model configurations with fallback chain including deepseek."""
        return {
            TaskType.ANALYTICS: [
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.1, 0.7, 300),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.1, 0.7, 200)
            ],
            TaskType.NARRATIVE: [
                ModelConfig(OLLAMA_NARRATIVE_MODEL, LLM_TIMEOUT_NARRATIVE, 0.3, 0.8, 500),
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.2, 0.7, 400),
                ModelConfig(OLLAMA_CHAT_MODEL, LLM_TIMEOUT_NARRATIVE, 0.2, 0.7, 400),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.1, 0.6, 300)
            ],
            TaskType.STRUCTURED: [
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.1, 0.6, 200),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.1, 0.6, 200)
            ],
            TaskType.CREATIVE: [
                ModelConfig(OLLAMA_NARRATIVE_MODEL, LLM_TIMEOUT_NARRATIVE, 0.4, 0.9, 600),
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.3, 0.8, 500),
                ModelConfig(OLLAMA_CHAT_MODEL, LLM_TIMEOUT_NARRATIVE, 0.3, 0.8, 500),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.2, 0.7, 400)
            ]
        }
    
    def get_response(self, prompt: str, task_type: TaskType, 
                    system_prompt: Optional[str] = None, max_retries: int = 2) -> Tuple[str, str]:
        """
        Get LLM response with progressive fallback and smart retry logic.
        Returns (response_text, model_used).
        """
        self.stats['total_requests'] += 1
        model_chain = self.models.get(task_type, self.models[TaskType.STRUCTURED])
        operation_name = f"{task_type.value}_generation"
        
        for i, model_config in enumerate(model_chain):
            # Try each model with retries
            for retry in range(max_retries):
                start_time = time.time()
                try:
                    response = self._make_request_with_validation(prompt, model_config, system_prompt, retry)
                    
                    # Update statistics
                    response_time = time.time() - start_time
                    self._update_stats(model_config.name, response_time, i > 0)
                    
                    # Record performance metrics (if available)
                    self._record_performance(operation_name, model_config.name, response_time, True)
                    
                    logger.debug(f"LLM response from {model_config.name} in {response_time:.2f}s (attempt {retry + 1})")
                    return response, model_config.name
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    error_msg = str(e)[:200]
                    
                    # Record failed attempt
                    self._record_performance(operation_name, model_config.name, response_time, False, error_msg)
                    
                    # Log detailed failure info for debugging
                    if task_type == TaskType.NARRATIVE:
                        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                        logger.warning(
                            f"Model {model_config.name} failed (attempt {retry + 1}/{max_retries}): {error_msg}\n"
                            f"Prompt preview: {prompt_preview}"
                        )
                    
                    if retry == max_retries - 1:
                        logger.warning(f"Model {model_config.name} failed all {max_retries} attempts")
                        break
                    
                    # Exponential backoff between retries
                    time.sleep(0.5 * (2 ** retry))
        
        logger.error(f"All models failed for {task_type} after {max_retries} attempts each")
        raise Exception(f"All LLM models failed for {task_type}")
    
    def _make_request_with_validation(self, prompt: str, config: ModelConfig, 
                                    system_prompt: Optional[str] = None, retry_count: int = 0) -> str:
        """Make a single LLM request with enhanced validation."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Adjust temperature slightly for retries to get different responses
        temperature = config.temperature + (retry_count * 0.1)
        temperature = min(temperature, 0.9)
        
        payload = {
            "model": config.name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                "stop": ["\n\n\n", "###", "---"]
            }
        }
        
        response = requests.post(self.chat_url, json=payload, timeout=config.timeout)
        response.raise_for_status()
        
        result = response.json()
        content = result.get('message', {}).get('content', '').strip()
        
        # Enhanced validation
        if not content:
            raise Exception("Empty response from model")
        
        if len(content) < 10:
            raise Exception(f"Response too short ({len(content)} chars): '{content}'")
        
        # Check for common failure patterns
        if content.lower() in ['sorry', 'i cannot', 'i can\'t', 'unable to']:
            raise Exception(f"Model declined to respond: '{content[:50]}...'")
        
        return content
    
    def _update_stats(self, model_name: str, response_time: float, was_fallback: bool):
        """Update performance statistics."""
        self.stats['successful_requests'] += 1
        if was_fallback:
            self.stats['fallbacks_used'] += 1
        
        # Update model usage
        if model_name not in self.stats['model_usage']:
            self.stats['model_usage'][model_name] = 0
        self.stats['model_usage'][model_name] += 1
        
        # Update average response time
        if model_name not in self.stats['avg_response_time']:
            self.stats['avg_response_time'][model_name] = []
        self.stats['avg_response_time'][model_name].append(response_time)
        
        # Keep only last 100 measurements
        if len(self.stats['avg_response_time'][model_name]) > 100:
            self.stats['avg_response_time'][model_name] = \
                self.stats['avg_response_time'][model_name][-100:]
    
    def get_analytics_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Get response optimized for analytics tasks."""
        return self.get_response(prompt, TaskType.ANALYTICS, system_prompt)
    
    def get_narrative_response(self, prompt: str, system_prompt: Optional[str] = None, max_retries: int = 2) -> Tuple[str, str]:
        """Get response optimized for narrative generation with enhanced reliability."""
        return self.get_response(prompt, TaskType.NARRATIVE, system_prompt, max_retries)
    
    def get_structured_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Get response optimized for structured data tasks."""
        return self.get_response(prompt, TaskType.STRUCTURED, system_prompt)
    
    def get_creative_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Get response optimized for creative tasks."""
        return self.get_response(prompt, TaskType.CREATIVE, system_prompt)
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        # Calculate average response times
        for model, times in stats['avg_response_time'].items():
            if times:
                stats['avg_response_time'][model] = sum(times) / len(times)
            else:
                stats['avg_response_time'][model] = 0
        
        # Calculate success rate
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['fallback_rate'] = stats['fallbacks_used'] / stats['total_requests']
        else:
            stats['success_rate'] = 0
            stats['fallback_rate'] = 0
        
        return stats
    
    def check_model_availability(self) -> Dict[str, bool]:
        """Check which models are available."""
        available = {}
        test_prompt = "Hello"
        
        all_models = set()
        for model_list in self.models.values():
            for config in model_list:
                all_models.add(config.name)
        
        for model_name in all_models:
            try:
                config = ModelConfig(model_name, 10, 0.1, 0.5, 50)
                self._make_request(test_prompt, config)
                available[model_name] = True
                logger.info(f"Model {model_name} is available")
            except Exception as e:
                available[model_name] = False
                logger.warning(f"Model {model_name} is not available: {e}")
        
        return available
    
    def _record_performance(self, operation_name: str, model_name: str, 
                          response_time: float, success: bool, error_msg: str = None):
        """Record performance metrics if monitor is available."""
        try:
            if self._performance_monitor is None:
                # Lazy import to avoid circular dependencies
                from utils.performance_monitor import performance_monitor
                self._performance_monitor = performance_monitor
            
            self._performance_monitor.record_operation(
                operation_name, model_name, response_time, success, error_msg
            )
        except ImportError:
            # Performance monitoring not available, continue without it
            pass
        except Exception as e:
            logger.debug(f"Performance recording failed: {e}")


# Global instance
llm_manager = LLMManager()

__all__ = ['LLMManager', 'TaskType', 'llm_manager']
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
    LLM_TIMEOUT_FALLBACK
)
from utils.performance_monitor import performance_monitor

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
    
    def _initialize_models(self) -> Dict[TaskType, List[ModelConfig]]:
        """Initialize model configurations with fallback chain."""
        return {
            TaskType.ANALYTICS: [
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.1, 0.7, 300),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.1, 0.7, 200)
            ],
            TaskType.NARRATIVE: [
                ModelConfig(OLLAMA_NARRATIVE_MODEL, LLM_TIMEOUT_NARRATIVE, 0.3, 0.8, 500),
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.2, 0.7, 400),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.1, 0.6, 300)
            ],
            TaskType.STRUCTURED: [
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.1, 0.6, 200),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.1, 0.6, 200)
            ],
            TaskType.CREATIVE: [
                ModelConfig(OLLAMA_NARRATIVE_MODEL, LLM_TIMEOUT_NARRATIVE, 0.4, 0.9, 600),
                ModelConfig(OLLAMA_ANALYTICS_MODEL, LLM_TIMEOUT_FAST, 0.3, 0.8, 500),
                ModelConfig(OLLAMA_FALLBACK_MODEL, LLM_TIMEOUT_FALLBACK, 0.2, 0.7, 400)
            ]
        }
    
    def get_response(self, prompt: str, task_type: TaskType, 
                    system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """
        Get LLM response with progressive fallback.
        Returns (response_text, model_used).
        """
        self.stats['total_requests'] += 1
        model_chain = self.models.get(task_type, self.models[TaskType.STRUCTURED])
        operation_name = f"{task_type.value}_generation"
        
        for i, model_config in enumerate(model_chain):
            start_time = time.time()
            try:
                response = self._make_request(prompt, model_config, system_prompt)
                
                # Update statistics
                response_time = time.time() - start_time
                self._update_stats(model_config.name, response_time, i > 0)
                
                # Record performance metrics
                performance_monitor.record_operation(
                    operation_name, model_config.name, response_time, True
                )
                
                logger.debug(f"LLM response from {model_config.name} in {response_time:.2f}s")
                return response, model_config.name
                
            except Exception as e:
                response_time = time.time() - start_time
                error_msg = str(e)[:200]  # Truncate long error messages
                
                # Record failed attempt
                performance_monitor.record_operation(
                    operation_name, model_config.name, response_time, False, error_msg
                )
                
                logger.warning(f"Model {model_config.name} failed: {error_msg}")
                if i == len(model_chain) - 1:  # Last model in chain
                    logger.error(f"All models failed for {task_type}")
                    raise Exception(f"All LLM models failed: {e}")
                continue
        
        raise Exception("No models available")
    
    def _make_request(self, prompt: str, config: ModelConfig, 
                     system_prompt: Optional[str] = None) -> str:
        """Make a single LLM request."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": config.name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_tokens,
                "stop": ["\n\n\n", "###", "---"]  # Stop sequences for cleaner output
            }
        }
        
        response = requests.post(self.chat_url, json=payload, timeout=config.timeout)
        response.raise_for_status()
        
        result = response.json()
        content = result.get('message', {}).get('content', '').strip()
        
        if not content:
            raise Exception("Empty response from model")
        
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
    
    def get_narrative_response(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Get response optimized for narrative generation."""
        return self.get_response(prompt, TaskType.NARRATIVE, system_prompt)
    
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


# Global instance
llm_manager = LLMManager()

__all__ = ['LLMManager', 'TaskType', 'llm_manager']
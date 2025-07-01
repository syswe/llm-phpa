"""
LLM Provider Module for Enhanced PHPA Benchmark

This module provides a unified interface for multiple LLM providers:
- Gemini 2.5 Pro (Google)
- Qwen3 (Alibaba)
- Grok-3 (xAI)

Each provider implements the same interface for consistent usage across the benchmark.
"""

import time
import requests
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    api_key: str
    model_name: str
    base_url: str
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, name: str, config: LLMConfig):
        self.name = name
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self):
        """Validate provider-specific configuration."""
        pass
    
    @abstractmethod
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """Prepare API request payload."""
        pass
    
    @abstractmethod
    def _extract_response(self, response_data: Dict) -> str:
        """Extract text response from API response."""
        pass
    
    def analyze(self, prompt: str) -> Optional[str]:
        """Send prompt to LLM and return response."""
        for attempt in range(self.config.max_retries):
            try:
                request_data = self._prepare_request(prompt)
                
                response = requests.post(
                    self.config.base_url,
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                return self._extract_response(response_data)
                
            except requests.exceptions.Timeout:
                logger.warning(f"{self.name} request timed out (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"{self.name} request failed: {e} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"{self.name} unexpected error: {e}")
                break
            
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay * (attempt + 1))
        
        logger.error(f"{self.name} failed after {self.config.max_retries} attempts")
        return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }

class GeminiProvider(LLMProvider):
    """Google Gemini 2.5 Pro provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__("Gemini 2.5 Pro", config)
    
    def _validate_config(self):
        """Validate Gemini configuration."""
        if not self.config.api_key or self.config.api_key == "YOUR_API_KEY":
            raise ValueError("Gemini API key not configured")
        
        if not self.config.base_url:
            self.config.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Gemini-specific headers."""
        return {'Content-Type': 'application/json'}
    
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """Prepare Gemini API request."""
        return {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": self.config.max_tokens
            }
        }
    
    def _extract_response(self, response_data: Dict) -> str:
        """Extract response from Gemini API."""
        if 'candidates' in response_data and response_data['candidates']:
            candidate = response_data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return "".join(
                    part['text'] for part in candidate['content']['parts'] 
                    if 'text' in part
                ).strip()
        
        logger.warning("Could not extract text from Gemini response")
        return ""
    
    def analyze(self, prompt: str) -> Optional[str]:
        """Override to handle Gemini's API key parameter."""
        for attempt in range(self.config.max_retries):
            try:
                request_data = self._prepare_request(prompt)
                
                response = requests.post(
                    f"{self.config.base_url}?key={self.config.api_key}",
                    headers=self._get_headers(),
                    json=request_data,
                    timeout=self.config.timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                return self._extract_response(response_data)
                
            except Exception as e:
                logger.warning(f"Gemini request failed: {e} (attempt {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        
        return None

class QwenProvider(LLMProvider):
    """Alibaba Qwen3 provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__("Qwen3", config)
    
    def _validate_config(self):
        """Validate Qwen configuration."""
        if not self.config.api_key:
            raise ValueError("Qwen API key not configured")
        
        if not self.config.base_url:
            self.config.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        
        if not self.config.model_name:
            self.config.model_name = "qwen-turbo"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Qwen-specific headers."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config.api_key}'
        }
    
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """Prepare Qwen API request."""
        return {
            "model": self.config.model_name,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": 0.8
            }
        }
    
    def _extract_response(self, response_data: Dict) -> str:
        """Extract response from Qwen API."""
        if 'output' in response_data and 'text' in response_data['output']:
            return response_data['output']['text'].strip()
        elif 'output' in response_data and 'choices' in response_data['output']:
            choices = response_data['output']['choices']
            if choices and 'message' in choices[0] and 'content' in choices[0]['message']:
                return choices[0]['message']['content'].strip()
        
        logger.warning("Could not extract text from Qwen response")
        return ""

class GrokProvider(LLMProvider):
    """xAI Grok-3 provider."""
    
    def __init__(self, config: LLMConfig):
        super().__init__("Grok-3", config)
    
    def _validate_config(self):
        """Validate Grok configuration."""
        if not self.config.api_key:
            raise ValueError("Grok API key not configured")
        
        if not self.config.base_url:
            self.config.base_url = "https://api.x.ai/v1/chat/completions"
        
        if not self.config.model_name:
            self.config.model_name = "grok-beta"
    
    def _prepare_request(self, prompt: str) -> Dict[str, Any]:
        """Prepare Grok API request."""
        return {
            "model": self.config.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": 0.8
        }
    
    def _extract_response(self, response_data: Dict) -> str:
        """Extract response from Grok API."""
        if 'choices' in response_data and response_data['choices']:
            choice = response_data['choices'][0]
            if 'message' in choice and 'content' in choice['message']:
                return choice['message']['content'].strip()
        
        logger.warning("Could not extract text from Grok response")
        return ""

class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    def __init__(self):
        self._providers = {
            'gemini': GeminiProvider,
            'qwen': QwenProvider,
            'grok': GrokProvider
        }
    
    def create_provider(self, provider_name: str, config) -> LLMProvider:
        """Create an LLM provider instance."""
        provider_name = provider_name.lower()
        
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        # Get provider-specific configuration
        llm_config = self._get_provider_config(provider_name, config)
        
        provider_class = self._providers[provider_name]
        return provider_class(llm_config)
    
    def _get_provider_config(self, provider_name: str, config) -> LLMConfig:
        """Get configuration for specific provider."""
        provider_config = getattr(config, f'{provider_name}_config', {})
        
        return LLMConfig(
            api_key=provider_config.get('api_key', ''),
            model_name=provider_config.get('model_name', ''),
            base_url=provider_config.get('base_url', ''),
            max_tokens=provider_config.get('max_tokens', 1000),
            temperature=provider_config.get('temperature', 0.1),
            timeout=provider_config.get('timeout', 60),
            max_retries=provider_config.get('max_retries', 3),
            retry_delay=provider_config.get('retry_delay', 2.0)
        )
    
    def list_providers(self) -> list:
        """List available providers."""
        return list(self._providers.keys())

# Utility functions for provider management
def get_available_providers() -> list:
    """Get list of available LLM providers."""
    factory = LLMProviderFactory()
    return factory.list_providers()

def test_provider_connection(provider: LLMProvider) -> bool:
    """Test if provider connection is working."""
    test_prompt = "Hello, this is a connection test. Please respond with 'OK'."
    
    try:
        response = provider.analyze(test_prompt)
        return response is not None and len(response.strip()) > 0
    except Exception as e:
        logger.error(f"Provider connection test failed: {e}")
        return False 
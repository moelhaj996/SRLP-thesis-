"""
Modular AI provider clients for GPT-4, Claude-3, and Gemini.
Each provider handles API calls with retry logic and error handling.
"""

import asyncio
import aiohttp
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ProviderResponse:
    """Standardized response from any AI provider."""
    content: str
    tokens_used: int
    cost_usd: float
    latency_ms: float
    model: str
    provider: str
    success: bool
    error: Optional[str] = None

@dataclass
class ProviderMetrics:
    """Metrics for provider performance tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    retry_count: int = 0
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.successful_requests, 1)
    
    @property
    def avg_cost_per_request(self) -> float:
        return self.total_cost / max(self.successful_requests, 1)

class BaseProvider(ABC):
    """Base class for all AI providers."""
    
    def __init__(self, api_key: str, max_retries: int = 5):
        self.api_key = api_key
        self.max_retries = max_retries
        self.metrics = ProviderMetrics()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> ProviderResponse:
        """Generate response from the provider."""
        pass
    
    async def _make_request_with_retry(self, request_func, *args, **kwargs) -> ProviderResponse:
        """Make request with exponential backoff retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.metrics.total_requests += 1
                
                start_time = time.time()
                response = await request_func(*args, **kwargs)
                latency = (time.time() - start_time) * 1000
                
                if response.success:
                    self.metrics.successful_requests += 1
                    self.metrics.total_tokens += response.tokens_used
                    self.metrics.total_cost += response.cost_usd
                    self.metrics.total_latency += latency
                    response.latency_ms = latency
                    return response
                else:
                    self.metrics.failed_requests += 1
                    if attempt < self.max_retries:
                        self.metrics.retry_count += 1
                        await self._wait_before_retry(attempt)
                        continue
                    else:
                        return response
                        
            except Exception as e:
                last_exception = e
                self.metrics.failed_requests += 1
                
                if attempt < self.max_retries:
                    self.metrics.retry_count += 1
                    await self._wait_before_retry(attempt)
                    continue
                else:
                    break
        
        # All retries failed
        return ProviderResponse(
            content="",
            tokens_used=0,
            latency_ms=0,
            cost_usd=0,
            model=self.model_name,
            provider=self.provider_name,
            success=False,
            error=str(last_exception) if last_exception else "Max retries exceeded"
        )
    
    async def _wait_before_retry(self, attempt: int):
        """Wait with exponential backoff and jitter."""
        base_delay = 2 ** attempt
        jitter = random.uniform(0.1, 0.3) * base_delay
        delay = min(base_delay + jitter, 60)  # Cap at 60 seconds
        await asyncio.sleep(delay)
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

class GPT4Provider(BaseProvider):
    """OpenAI GPT-4 provider implementation."""
    
    def __init__(self, api_key: str, max_retries: int = 5):
        super().__init__(api_key, max_retries)
        self.base_url = "https://api.openai.com/v1/chat/completions"
        # Updated pricing for GPT-4 Turbo
        self.input_cost_per_1k = 0.01   # $0.01 per 1K input tokens
        self.output_cost_per_1k = 0.03  # $0.03 per 1K output tokens
    
    @property
    def model_name(self) -> str:
        return "gpt-4-turbo-preview"
    
    @property
    def provider_name(self) -> str:
        return "gpt4"
    
    async def generate_response(self, prompt: str, **kwargs) -> ProviderResponse:
        """Generate response from GPT-4."""
        return await self._make_request_with_retry(self._make_openai_request, prompt, **kwargs)
    
    async def _make_openai_request(self, prompt: str, **kwargs) -> ProviderResponse:
        """Make request to OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 3000),
            "temperature": kwargs.get("temperature", 0.7),
            "seed": kwargs.get("seed", 42)  # For determinism
        }
        
        try:
            async with self.session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_openai_response(data)
                else:
                    error_text = await response.text()
                    return ProviderResponse(
                        content="",
                        tokens_used=0,
                        latency_ms=0,
                        cost_usd=0,
                        model=self.model_name,
                        provider=self.provider_name,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            return ProviderResponse(
                content="",
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                model=self.model_name,
                provider=self.provider_name,
                success=False,
                error=str(e)
            )
    
    def _parse_openai_response(self, data: Dict[str, Any]) -> ProviderResponse:
        """Parse OpenAI API response."""
        try:
            content = data["choices"][0]["message"]["content"]
            usage = data["usage"]
            
            input_tokens = usage["prompt_tokens"]
            output_tokens = usage["completion_tokens"]
            total_tokens = usage["total_tokens"]
            
            # Calculate cost
            cost = (input_tokens * self.input_cost_per_1k / 1000 + 
                   output_tokens * self.output_cost_per_1k / 1000)
            
            return ProviderResponse(
                content=content,
                tokens_used=total_tokens,
                latency_ms=0,  # Will be set by caller
                cost_usd=cost,
                model=self.model_name,
                provider=self.provider_name,
                success=True
            )
        except KeyError as e:
            return ProviderResponse(
                content="",
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                model=self.model_name,
                provider=self.provider_name,
                success=False,
                error=f"Failed to parse response: {str(e)}"
            )

class Claude3Provider(BaseProvider):
    """Anthropic Claude-3 provider implementation."""
    
    def __init__(self, api_key: str, max_retries: int = 5):
        super().__init__(api_key, max_retries)
        self.base_url = "https://api.anthropic.com/v1/messages"
        # Claude-3 Sonnet pricing
        self.input_cost_per_1k = 0.003   # $0.003 per 1K input tokens
        self.output_cost_per_1k = 0.015  # $0.015 per 1K output tokens
    
    @property
    def model_name(self) -> str:
        return "claude-3-sonnet-20240229"
    
    @property
    def provider_name(self) -> str:
        return "claude3"
    
    async def generate_response(self, prompt: str, **kwargs) -> ProviderResponse:
        """Generate response from Claude-3."""
        return await self._make_request_with_retry(self._make_anthropic_request, prompt, **kwargs)
    
    async def _make_anthropic_request(self, prompt: str, **kwargs) -> ProviderResponse:
        """Make request to Anthropic API."""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 3000),
            "temperature": kwargs.get("temperature", 0.7),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            async with self.session.post(self.base_url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_anthropic_response(data)
                else:
                    error_text = await response.text()
                    return ProviderResponse(
                        content="",
                        tokens_used=0,
                        latency_ms=0,
                        cost_usd=0,
                        model=self.model_name,
                        provider=self.provider_name,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            return ProviderResponse(
                content="",
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                model=self.model_name,
                provider=self.provider_name,
                success=False,
                error=str(e)
            )
    
    def _parse_anthropic_response(self, data: Dict[str, Any]) -> ProviderResponse:
        """Parse Anthropic API response."""
        try:
            content = data["content"][0]["text"]
            usage = data["usage"]
            
            input_tokens = usage["input_tokens"]
            output_tokens = usage["output_tokens"]
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            cost = (input_tokens * self.input_cost_per_1k / 1000 + 
                   output_tokens * self.output_cost_per_1k / 1000)
            
            return ProviderResponse(
                content=content,
                tokens_used=total_tokens,
                latency_ms=0,  # Will be set by caller
                cost_usd=cost,
                model=self.model_name,
                provider=self.provider_name,
                success=True
            )
        except KeyError as e:
            return ProviderResponse(
                content="",
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                model=self.model_name,
                provider=self.provider_name,
                success=False,
                error=f"Failed to parse response: {str(e)}"
            )

class GeminiProvider(BaseProvider):
    """Google Gemini provider implementation."""
    
    def __init__(self, api_key: str, max_retries: int = 5):
        super().__init__(api_key, max_retries)
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        # Gemini Pro pricing (estimated)
        self.input_cost_per_1k = 0.001   # $0.001 per 1K input tokens
        self.output_cost_per_1k = 0.002  # $0.002 per 1K output tokens
    
    @property
    def model_name(self) -> str:
        return "gemini-pro"
    
    @property
    def provider_name(self) -> str:
        return "gemini"
    
    async def generate_response(self, prompt: str, **kwargs) -> ProviderResponse:
        """Generate response from Gemini."""
        return await self._make_request_with_retry(self._make_gemini_request, prompt, **kwargs)
    
    async def _make_gemini_request(self, prompt: str, **kwargs) -> ProviderResponse:
        """Make request to Gemini API."""
        url = f"{self.base_url}?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 3000),
                "candidateCount": 1
            }
        }
        
        try:
            async with self.session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_gemini_response(data)
                else:
                    error_text = await response.text()
                    return ProviderResponse(
                        content="",
                        tokens_used=0,
                        latency_ms=0,
                        cost_usd=0,
                        model=self.model_name,
                        provider=self.provider_name,
                        success=False,
                        error=f"HTTP {response.status}: {error_text}"
                    )
        except Exception as e:
            return ProviderResponse(
                content="",
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                model=self.model_name,
                provider=self.provider_name,
                success=False,
                error=str(e)
            )
    
    def _parse_gemini_response(self, data: Dict[str, Any]) -> ProviderResponse:
        """Parse Gemini API response."""
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Estimate token usage (Gemini doesn't always provide exact counts)
            estimated_input_tokens = len(content.split()) * 0.8  # Rough estimation
            estimated_output_tokens = len(content.split()) * 1.2
            estimated_total_tokens = int(estimated_input_tokens + estimated_output_tokens)
            
            # Calculate cost
            cost = (estimated_input_tokens * self.input_cost_per_1k / 1000 + 
                   estimated_output_tokens * self.output_cost_per_1k / 1000)
            
            return ProviderResponse(
                content=content,
                tokens_used=estimated_total_tokens,
                latency_ms=0,  # Will be set by caller
                cost_usd=cost,
                model=self.model_name,
                provider=self.provider_name,
                success=True
            )
        except (KeyError, IndexError) as e:
            return ProviderResponse(
                content="",
                tokens_used=0,
                latency_ms=0,
                cost_usd=0,
                model=self.model_name,
                provider=self.provider_name,
                success=False,
                error=f"Failed to parse response: {str(e)}"
            )

class ProviderManager:
    """Manages multiple AI providers."""
    
    def __init__(self, config):
        self.config = config
        self.providers: Dict[str, BaseProvider] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all providers based on config."""
        provider_map = {
            "gpt4": (GPT4Provider, self.config.openai_api_key),
            "claude3": (Claude3Provider, self.config.anthropic_api_key),
            "gemini": (GeminiProvider, self.config.gemini_api_key)
        }
        
        for provider_name in self.config.providers:
            if provider_name in provider_map:
                provider_class, api_key = provider_map[provider_name]
                self.providers[provider_name] = provider_class(api_key, self.config.max_retries)
    
    async def __aenter__(self):
        for provider in self.providers.values():
            await provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for provider in self.providers.values():
            await provider.__aexit__(exc_type, exc_val, exc_tb)
    
    def get_provider(self, provider_name: str) -> BaseProvider:
        """Get provider by name."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers[provider_name]
    
    def get_all_metrics(self) -> Dict[str, ProviderMetrics]:
        """Get metrics for all providers."""
        return {name: provider.metrics for name, provider in self.providers.items()}

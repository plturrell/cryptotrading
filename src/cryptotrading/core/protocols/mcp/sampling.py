"""
MCP Sampling Implementation
Implements the sampling/completion API for LLM integration
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class SamplingMessage:
    """Message in sampling conversation"""

    role: str  # "user", "assistant", "system"
    content: Union[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        if isinstance(self.content, str):
            return {"role": self.role, "content": {"type": "text", "text": self.content}}
        return {"role": self.role, "content": self.content}


@dataclass
class SamplingRequest:
    """Sampling request parameters"""

    messages: List[SamplingMessage]
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    include_thinking: Optional[bool] = None
    model_preferences: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        result = {"messages": [msg.to_dict() for msg in self.messages]}

        if self.system_prompt:
            result["systemPrompt"] = self.system_prompt
        if self.max_tokens:
            result["maxTokens"] = self.max_tokens
        if self.temperature is not None:
            result["temperature"] = self.temperature
        if self.top_p is not None:
            result["topP"] = self.top_p
        if self.stop_sequences:
            result["stopSequences"] = self.stop_sequences
        if self.include_thinking is not None:
            result["includeThinking"] = self.include_thinking
        if self.model_preferences:
            result["modelPreferences"] = self.model_preferences

        return result


@dataclass
class SamplingResponse:
    """Sampling response from LLM"""

    role: str
    content: Union[str, Dict[str, Any]]
    model: Optional[str] = None
    stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        if isinstance(self.content, str):
            result = {"role": self.role, "content": {"type": "text", "text": self.content}}
        else:
            result = {"role": self.role, "content": self.content}

        if self.model:
            result["model"] = self.model
        if self.stop_reason:
            result["stopReason"] = self.stop_reason

        return result


class SamplingProvider:
    """Base class for sampling providers"""

    def __init__(self, name: str):
        self.name = name

    async def create_message(self, request: SamplingRequest) -> SamplingResponse:
        """Create a message response"""
        raise NotImplementedError

    async def create_message_stream(
        self, request: SamplingRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create a streaming message response"""
        # Default implementation: just yield the final result
        response = await self.create_message(request)
        yield {"type": "content", "content": response.content}


class MockSamplingProvider(SamplingProvider):
    """Mock sampling provider for testing"""

    def __init__(self):
        super().__init__("mock")

    async def create_message(self, request: SamplingRequest) -> SamplingResponse:
        """Create mock response"""
        # Simulate processing delay
        await asyncio.sleep(0.1)

        # Generate mock response based on last message
        last_message = request.messages[-1] if request.messages else None

        if last_message and isinstance(last_message.content, str):
            content = f"Mock response to: {last_message.content[:100]}..."
        elif last_message and isinstance(last_message.content, dict):
            text = last_message.content.get("text", "")
            content = f"Mock response to: {text[:100]}..."
        else:
            content = "Mock response: Hello! I'm a mock sampling provider."

        return SamplingResponse(
            role="assistant", content=content, model="mock-model-1.0", stop_reason="max_tokens"
        )

    async def create_message_stream(
        self, request: SamplingRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create streaming mock response"""
        response = await self.create_message(request)
        content = response.content

        # Stream word by word
        if isinstance(content, str):
            words = content.split()
            for i, word in enumerate(words):
                await asyncio.sleep(0.01)  # Simulate streaming delay

                if i == 0:
                    yield {"type": "content", "content": {"type": "text", "text": word}}
                else:
                    yield {"type": "content", "content": {"type": "text", "text": f" {word}"}}

        # Final message
        yield {"type": "message", "message": response.to_dict()}


class OpenAISamplingProvider(SamplingProvider):
    """OpenAI API sampling provider"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__("openai")
        self.api_key = api_key
        self.model = model

    async def create_message(self, request: SamplingRequest) -> SamplingResponse:
        """Create message using OpenAI API"""
        try:
            import openai
        except ImportError:
            raise RuntimeError("OpenAI library not installed. Install with: pip install openai")

        # Convert messages to OpenAI format
        messages = []

        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            content = msg.content
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            elif isinstance(content, dict):
                content = json.dumps(content)

            messages.append({"role": msg.role, "content": content})

        # Create OpenAI client
        client = openai.AsyncOpenAI(api_key=self.api_key)

        # Prepare parameters
        params = {"model": self.model, "messages": messages}

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop"] = request.stop_sequences

        # Make API call
        response = await client.chat.completions.create(**params)

        choice = response.choices[0]
        return SamplingResponse(
            role="assistant",
            content=choice.message.content,
            model=response.model,
            stop_reason=choice.finish_reason,
        )

    async def create_message_stream(
        self, request: SamplingRequest
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create streaming message using OpenAI API"""
        try:
            import openai
        except ImportError:
            raise RuntimeError("OpenAI library not installed")

        # Convert messages (same as above)
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            content = msg.content
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            elif isinstance(content, dict):
                content = json.dumps(content)

            messages.append({"role": msg.role, "content": content})

        client = openai.AsyncOpenAI(api_key=self.api_key)

        params = {"model": self.model, "messages": messages, "stream": True}

        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop"] = request.stop_sequences

        # Stream response
        accumulated_content = ""
        async for chunk in await client.chat.completions.create(**params):
            choice = chunk.choices[0]

            if choice.delta.content:
                accumulated_content += choice.delta.content
                yield {"type": "content", "content": {"type": "text", "text": choice.delta.content}}

            if choice.finish_reason:
                yield {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": {"type": "text", "text": accumulated_content},
                        "model": chunk.model,
                        "stopReason": choice.finish_reason,
                    },
                }


class AnthropicSamplingProvider(SamplingProvider):
    """Anthropic Claude sampling provider"""

    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__("anthropic")
        self.api_key = api_key
        self.model = model

    async def create_message(self, request: SamplingRequest) -> SamplingResponse:
        """Create message using Anthropic API"""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "Anthropic library not installed. Install with: pip install anthropic"
            )

        # Convert messages to Anthropic format
        messages = []
        for msg in request.messages:
            content = msg.content
            if isinstance(content, dict) and "text" in content:
                content = content["text"]
            elif isinstance(content, dict):
                content = json.dumps(content)

            messages.append({"role": msg.role, "content": content})

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        params = {"model": self.model, "messages": messages}

        if request.system_prompt:
            params["system"] = request.system_prompt
        if request.max_tokens:
            params["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.top_p is not None:
            params["top_p"] = request.top_p
        if request.stop_sequences:
            params["stop_sequences"] = request.stop_sequences

        response = await client.messages.create(**params)

        return SamplingResponse(
            role="assistant",
            content=response.content[0].text,
            model=response.model,
            stop_reason=response.stop_reason,
        )


class SamplingManager:
    """Manager for sampling providers"""

    def __init__(self):
        self.providers: Dict[str, SamplingProvider] = {}
        self.default_provider: Optional[str] = None

    def register_provider(self, provider: SamplingProvider, set_as_default: bool = False):
        """Register a sampling provider"""
        self.providers[provider.name] = provider
        logger.info(f"Registered sampling provider: {provider.name}")

        if set_as_default or not self.default_provider:
            self.default_provider = provider.name

    def get_provider(self, name: Optional[str] = None) -> Optional[SamplingProvider]:
        """Get provider by name or default"""
        if name:
            return self.providers.get(name)
        elif self.default_provider:
            return self.providers.get(self.default_provider)
        return None

    async def create_message(
        self, request: SamplingRequest, provider_name: Optional[str] = None
    ) -> SamplingResponse:
        """Create message using specified or default provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            available = list(self.providers.keys())
            raise ValueError(f"No sampling provider available. Registered: {available}")

        return await provider.create_message(request)

    async def create_message_stream(
        self, request: SamplingRequest, provider_name: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create streaming message using specified or default provider"""
        provider = self.get_provider(provider_name)
        if not provider:
            available = list(self.providers.keys())
            raise ValueError(f"No sampling provider available. Registered: {available}")

        async for chunk in provider.create_message_stream(request):
            yield chunk


# Global sampling manager
sampling_manager = SamplingManager()

# Register mock provider by default
sampling_manager.register_provider(MockSamplingProvider(), set_as_default=True)


# Helper functions
def parse_sampling_request(params: Dict[str, Any]) -> SamplingRequest:
    """Parse MCP sampling request parameters"""
    messages = []
    for msg_data in params.get("messages", []):
        content = msg_data.get("content", "")
        if isinstance(content, dict) and "text" in content:
            content = content["text"]

        messages.append(SamplingMessage(role=msg_data["role"], content=content))

    return SamplingRequest(
        messages=messages,
        system_prompt=params.get("systemPrompt"),
        max_tokens=params.get("maxTokens"),
        temperature=params.get("temperature"),
        top_p=params.get("topP"),
        stop_sequences=params.get("stopSequences"),
        include_thinking=params.get("includeThinking"),
        model_preferences=params.get("modelPreferences"),
    )


def setup_sampling_providers(config: Dict[str, Any]):
    """Setup sampling providers from configuration"""
    for provider_config in config.get("providers", []):
        provider_type = provider_config.get("type")

        if provider_type == "openai":
            provider = OpenAISamplingProvider(
                api_key=provider_config["api_key"], model=provider_config.get("model", "gpt-4")
            )
        elif provider_type == "anthropic":
            provider = AnthropicSamplingProvider(
                api_key=provider_config["api_key"],
                model=provider_config.get("model", "claude-3-sonnet-20240229"),
            )
        elif provider_type == "mock":
            provider = MockSamplingProvider()
        else:
            logger.warning(f"Unknown sampling provider type: {provider_type}")
            continue

        sampling_manager.register_provider(
            provider, set_as_default=provider_config.get("default", False)
        )

"""
Base Model class for Strands
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterable, Type

class Model(ABC):
    """Abstract base class for all models in strands"""
    
    @abstractmethod
    def get_config(self) -> Any:
        """Get model configuration"""
        pass
    
    @abstractmethod
    def update_config(self, **model_config: Any) -> None:
        """Update model configuration"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Any],
        tool_specs: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[Any]:
        """Stream model response"""
        pass
    
    @abstractmethod
    async def structured_output(
        self,
        output_model: Type,
        prompt: List[Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterable[Dict[str, Any]]:
        """Generate structured output"""
        pass
"""
MCP Prompts Implementation
Implements the prompts API for structured prompt management
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class PromptArgument:
    """Prompt argument definition"""

    name: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        result = {"name": self.name, "required": self.required}
        if self.description:
            result["description"] = self.description
        if self.default is not None:
            result["default"] = self.default
        return result


@dataclass
class PromptMessage:
    """Prompt message in conversation"""

    role: str  # "user", "assistant", "system"
    content: Union[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP format"""
        if isinstance(self.content, str):
            return {"role": self.role, "content": {"type": "text", "text": self.content}}
        return {"role": self.role, "content": self.content}


class MCPPrompt:
    """MCP Prompt definition"""

    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        arguments: Optional[List[PromptArgument]] = None,
        template: Optional[str] = None,
        handler: Optional[Callable] = None,
    ):
        self.name = name
        self.description = description
        self.arguments = arguments or []
        self.template = template
        self.handler = handler
        self._messages_cache: Dict[str, List[PromptMessage]] = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to MCP prompt format"""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": [arg.to_dict() for arg in self.arguments],
        }

    async def get_messages(self, arguments: Dict[str, Any]) -> List[PromptMessage]:
        """Get prompt messages with arguments filled in"""
        # Validate required arguments
        for arg in self.arguments:
            if arg.required and arg.name not in arguments:
                raise ValueError(f"Missing required argument: {arg.name}")

        # Apply defaults
        filled_args = {}
        for arg in self.arguments:
            if arg.name in arguments:
                filled_args[arg.name] = arguments[arg.name]
            elif arg.default is not None:
                filled_args[arg.name] = arg.default

        # Generate messages
        if self.handler:
            # Custom handler
            messages = await self.handler(**filled_args)
            if isinstance(messages, str):
                messages = [PromptMessage("user", messages)]
            elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
                messages = [PromptMessage(m["role"], m["content"]) for m in messages]
        elif self.template:
            # Template-based
            try:
                content = self.template.format(**filled_args)
                messages = [PromptMessage("user", content)]
            except KeyError as e:
                raise ValueError(f"Template references undefined argument: {e}")
        else:
            # Default message
            messages = [PromptMessage("user", f"Execute {self.name} with {filled_args}")]

        return messages


class PromptRegistry:
    """Registry for managing prompts"""

    def __init__(self):
        self.prompts: Dict[str, MCPPrompt] = {}
        self._change_handlers: List[Callable] = []

    def register(self, prompt: MCPPrompt):
        """Register a prompt"""
        self.prompts[prompt.name] = prompt
        logger.info(f"Registered prompt: {prompt.name}")
        self._notify_change()

    def unregister(self, name: str):
        """Unregister a prompt"""
        if name in self.prompts:
            del self.prompts[name]
            logger.info(f"Unregistered prompt: {name}")
            self._notify_change()

    def get(self, name: str) -> Optional[MCPPrompt]:
        """Get prompt by name"""
        return self.prompts.get(name)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all prompts in MCP format"""
        return [prompt.to_dict() for prompt in self.prompts.values()]

    async def get_prompt_messages(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get prompt messages"""
        prompt = self.get(name)
        if not prompt:
            raise ValueError(f"Prompt not found: {name}")

        messages = await prompt.get_messages(arguments or {})
        return [msg.to_dict() for msg in messages]

    def on_change(self, handler: Callable):
        """Register change handler"""
        self._change_handlers.append(handler)

    def _notify_change(self):
        """Notify handlers of prompt list change"""
        for handler in self._change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler())
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in prompt change handler: {e}")


# Global prompt registry
prompt_registry = PromptRegistry()


# Built-in prompts
def register_builtin_prompts():
    """Register built-in prompts"""

    # Code analysis prompt
    code_analysis = MCPPrompt(
        name="analyze_code",
        description="Analyze code for potential issues and improvements",
        arguments=[
            PromptArgument("language", "Programming language", required=True),
            PromptArgument("code", "Code to analyze", required=True),
            PromptArgument("focus", "Specific aspect to focus on", required=False),
        ],
        template="""Please analyze the following {language} code:

```{language}
{code}
```

{focus}

Provide insights on:
1. Potential bugs or issues
2. Performance considerations
3. Best practices
4. Security concerns
5. Suggested improvements""",
    )

    # Documentation generation prompt
    doc_generation = MCPPrompt(
        name="generate_docs",
        description="Generate documentation for code",
        arguments=[
            PromptArgument("code", "Code to document", required=True),
            PromptArgument("style", "Documentation style", default="markdown"),
            PromptArgument("detail_level", "Level of detail", default="standard"),
        ],
        template="""Generate {style} documentation for the following code:

```
{code}
```

Detail level: {detail_level}

Include:
- Purpose and overview
- Parameters/arguments
- Return values
- Usage examples
- Any important notes or warnings""",
    )

    # Error explanation prompt
    async def explain_error_handler(error_message: str, context: Optional[str] = None, **kwargs):
        """Custom handler for error explanation"""
        messages = [
            PromptMessage(
                "system", "You are a helpful assistant that explains programming errors clearly."
            ),
            PromptMessage("user", f"Please explain this error:\n\n{error_message}"),
        ]

        if context:
            messages.append(PromptMessage("user", f"Context:\n{context}"))

        return messages

    error_explanation = MCPPrompt(
        name="explain_error",
        description="Explain an error message in detail",
        arguments=[
            PromptArgument("error_message", "The error message", required=True),
            PromptArgument("context", "Additional context", required=False),
        ],
        handler=explain_error_handler,
    )

    # Register all built-in prompts
    for prompt in [code_analysis, doc_generation, error_explanation]:
        prompt_registry.register(prompt)


# Initialize built-in prompts on module load
register_builtin_prompts()


# Helper functions for creating prompts
def create_prompt(
    name: str, description: str, template: str, arguments: List[Dict[str, Any]]
) -> MCPPrompt:
    """Create a prompt from configuration"""
    prompt_args = []
    for arg in arguments:
        prompt_args.append(
            PromptArgument(
                name=arg["name"],
                description=arg.get("description"),
                required=arg.get("required", False),
                default=arg.get("default"),
            )
        )

    return MCPPrompt(name=name, description=description, arguments=prompt_args, template=template)


def create_dynamic_prompt(name: str, description: str, handler: Callable) -> MCPPrompt:
    """Create a dynamic prompt with custom handler"""
    # Extract arguments from handler signature
    import inspect

    sig = inspect.signature(handler)

    arguments = []
    for param_name, param in sig.parameters.items():
        if param_name in ["kwargs", "args"]:
            continue

        required = param.default == inspect.Parameter.empty
        arguments.append(
            PromptArgument(
                name=param_name, required=required, default=None if required else param.default
            )
        )

    return MCPPrompt(name=name, description=description, arguments=arguments, handler=handler)

from .anthropic import AnthropicRunner
from .openai import OpenAIRunner

RUNNERS = {
    "claude-opus-4-6": AnthropicRunner,
    "gpt-5.4": OpenAIRunner,
}

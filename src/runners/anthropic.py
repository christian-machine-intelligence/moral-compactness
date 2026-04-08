"""Anthropic API runner with extended thinking support."""

import os
from anthropic import AsyncAnthropic


class AnthropicRunner:
    MODEL = "claude-opus-4-6"
    THINKING_BUDGET = 10000

    def __init__(self):
        self.client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def run(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 16000,
    ) -> dict:
        """Run a conversation and return response + thinking content.

        Returns dict with keys:
            response: str - the assistant's visible response
            thinking: str | None - extended thinking content
            tool_calls: list[dict] - any tool calls the model made
            error: str | None
        """
        try:
            resp = await self.client.messages.create(
                model=self.MODEL,
                max_tokens=max_tokens,
                temperature=1.0,  # required when thinking is enabled
                thinking={"type": "adaptive"},
                system=system_prompt,
                messages=messages,
            )

            thinking_text = ""
            response_text = ""
            tool_calls = []

            for block in resp.content:
                if block.type == "thinking":
                    thinking_text += block.thinking + "\n"
                elif block.type == "text":
                    response_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                    })

            return {
                "response": response_text.strip(),
                "thinking": thinking_text.strip() or None,
                "tool_calls": tool_calls,
                "error": None,
            }

        except Exception as e:
            return {
                "response": None,
                "thinking": None,
                "tool_calls": [],
                "error": str(e),
            }

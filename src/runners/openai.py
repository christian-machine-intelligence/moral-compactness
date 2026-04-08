"""OpenAI API runner with reasoning support for GPT 5.4."""

import os
from openai import AsyncOpenAI


class OpenAIRunner:
    MODEL = "gpt-5.4"

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    async def run(
        self,
        system_prompt: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Run a conversation and return response + reasoning content.

        Returns dict with keys:
            response: str - the assistant's visible response
            thinking: str | None - reasoning content (if available)
            tool_calls: list[dict] - any tool calls the model made
            error: str | None
        """
        try:
            api_messages = [{"role": "developer", "content": system_prompt}]
            api_messages.extend(messages)

            resp = await self.client.responses.create(
                model=self.MODEL,
                input=api_messages,
                reasoning={"effort": "high", "summary": "auto"},
                max_output_tokens=max_tokens,
            )

            thinking_text = ""
            response_text = ""
            tool_calls = []

            for item in resp.output:
                if item.type == "reasoning":
                    for part in item.summary:
                        thinking_text += part.text + "\n"
                elif item.type == "message":
                    for part in item.content:
                        if part.type == "output_text":
                            response_text += part.text
                elif item.type == "function_call":
                    tool_calls.append({
                        "name": item.name,
                        "input": item.arguments,
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

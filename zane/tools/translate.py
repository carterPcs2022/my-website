"""LLM-backed translation tool, wired into ToolRegistry as an LLM function
call. Uses the project's AsyncGroqClient (see zane/groq_client.py) so
translation requests share the same retry/backoff and rate-limit handling
as every other Groq call in the system.
"""
from __future__ import annotations

from typing import Any, Dict

from zane.groq_client import AsyncGroqClient

TRANSLATE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "translate_text",
        "description": (
            "Translates a piece of text into a target language. Use this "
            "when the user explicitly asks for a translation or asks what "
            "something means in another language."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to translate.",
                },
                "target_language": {
                    "type": "string",
                    "description": (
                        "The language to translate the text into "
                        "(e.g. 'French', 'Japanese')."
                    ),
                },
            },
            "required": ["text", "target_language"],
        },
    },
}


async def translate_text(text: str, target_language: str, groq_client: AsyncGroqClient) -> str:
    """Real translation via LLM — no flavor caveat needed, this genuinely works."""
    completion = await groq_client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate the following text to {target_language}. "
                    "Return only the translation, no explanation."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return completion.choices[0].message.content or ""

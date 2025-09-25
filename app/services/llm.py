# app/services/llm.py
from __future__ import annotations
import os
from typing import Generator
from . import config

def _build_messages(system: str, context: str, user_query: str):
    """
    Build messages for the LLM with explicit instructions:
    - Prioritize provided context (≈90%)
    - Allow limited external knowledge (≈10%) only if needed
    - No inline citations, page numbers, or standard names in the answer text
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": (
            f"Context (primary knowledge, ~90% priority):\n{context}\n\n"
            f"User question: {user_query}\n\n"
            "Answer the question with a concise, professional, and useful explanation. "
            "Rely primarily on the provided context data (~90%), but you may use limited external "
            "domain knowledge (~10%) to improve clarity or fill gaps if the context is insufficient. "
            "Make sure the answer is accurate, practical, and directly helpful for lifting and rigging operations.\n\n"
            "Rules:\n"
            "- DO NOT include inline citations, page numbers, or standard names in the answer text.\n"
            "- DO NOT hallucinate standards, documents, or URLs.\n"
            "- If information is missing, provide general best practices instead of making something up.\n"
            "- Keep the tone clear, professional, and focused."
        )}
    ]

def generate(system: str, context: str, user_query: str) -> str:
    """
    Generate a non-streamed response using the configured LLM provider.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "Missing package 'openai'. Install with: pip install 'openai>=1.0.0'"
        ) from e

    api_key = config.ENV.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in environment/.env")

    client = OpenAI(api_key=api_key)
    messages = _build_messages(system, context, user_query)

    resp = client.chat.completions.create(
        model=config.ENV.OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()

def stream_chat(system: str, context: str, user_query: str) -> Generator[str, None, None]:
    """
    Stream the response token-by-token using the configured LLM provider.
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "Missing package 'openai'. Install with: pip install 'openai>=1.0.0'"
        ) from e

    api_key = config.ENV.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing in environment/.env")

    client = OpenAI(api_key=api_key)
    messages = _build_messages(system, context, user_query)

    stream = client.chat.completions.create(
        model=config.ENV.OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            yield delta

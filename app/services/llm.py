# app/services/llm.py
from __future__ import annotations
import os
from typing import Generator
from . import config

def _build_messages(system: str, context: str, user_query: str):
    return [
        {"role": "system", "content": system},
        {"role": "user",
         "content": (
             f"Context:\n{context}\n\n"
             f"Question:\n{user_query}\n\n"
             "Follow the system rules strictly. Cite standards and page numbers where applicable."
         )},
    ]

def generate(system: str, context: str, user_query: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing package 'openai'. Install with: pip install 'openai>=1.0.0'") from e

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
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing package 'openai'. Install with: pip install 'openai>=1.0.0'") from e

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

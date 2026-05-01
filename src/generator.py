"""Local LLM generation via Ollama HTTP API.

Strict grounding prompt:
- The model is told to answer ONLY from the provided context.
- If context is insufficient, it must reply with literally `I don't know`.

Small instruction-tuned models (e.g. llama3.2:3b) sometimes produce a real
answer AND THEN reflexively append the safety phrase. We post-process the
output to strip a trailing "I don't know" when the response also contains
real content — leaving the genuine refusal case untouched.
"""
from __future__ import annotations

import json
import re
from typing import Iterator

import requests

from config import LLM_MODEL, OLLAMA_HOST
from src.vector_store import Hit


_TRAIL_IDK = re.compile(
    r"\s*\bI\s*don['’]?t\s*know\.?\s*$",
    re.IGNORECASE,
)


def clean_answer(text: str) -> str:
    """Remove a trailing 'I don't know' tag-along when a real answer precedes it."""
    if not text:
        return text
    stripped = text.strip()
    # Pure refusal — keep as-is (canonicalize punctuation/case).
    if re.fullmatch(r"\s*I\s*don['’]?t\s*know\.?\s*", stripped, flags=re.IGNORECASE):
        return "I don't know."
    cleaned = _TRAIL_IDK.sub("", stripped).rstrip()
    return cleaned or "I don't know."


SYSTEM_PROMPT = (
    "You are a factual assistant. Answer the user's question using ONLY the "
    "information in the CONTEXT below.\n"
    "Rules:\n"
    "- If, AND ONLY IF, the context does not contain enough information to "
    "answer at all, your ENTIRE response must be exactly the four words: "
    "I don't know.\n"
    "- If you do answer, write the answer and STOP. Never append "
    "\"I don't know\" after a real answer. Never hedge with a follow-up "
    "disclaimer.\n"
    "- Be concise. Do not invent facts. Do not mention the context — just "
    "answer.\n"
    "- For comparison questions, produce a short side-by-side comparison and "
    "stop."
)


def _format_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        title = (h.metadata or {}).get("title", "Unknown")
        kind = (h.metadata or {}).get("type", "?")
        blocks.append(f"[{i}] ({kind}: {title})\n{h.text}")
    return "\n\n".join(blocks)


def build_prompt(query: str, hits: list[Hit]) -> str:
    context = _format_context(hits) if hits else "(no context retrieved)"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER:"
    )


def generate(query: str, hits: list[Hit], model: str = LLM_MODEL,
             host: str = OLLAMA_HOST, temperature: float = 0.2) -> str:
    prompt = build_prompt(query, hits)
    r = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    r.raise_for_status()
    return clean_answer(r.json().get("response", ""))


def generate_stream(query: str, hits: list[Hit], model: str = LLM_MODEL,
                    host: str = OLLAMA_HOST, temperature: float = 0.2) -> Iterator[str]:
    prompt = build_prompt(query, hits)
    with requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        },
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError:
                continue
            chunk = obj.get("response", "")
            if chunk:
                yield chunk
            if obj.get("done"):
                break

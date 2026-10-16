"""LLM layer: talks to any OpenAI-compatible endpoint with nothing but urllib.

Default target is Gemini's OpenAI-compatible API, so the same code also works
with Ollama / LM Studio / OpenRouter by changing `base_url` and `model`.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("glucotwin.llm")

GEMINI_BASE_URL = os.getenv("GLUCOTWIN_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
DEFAULT_MODEL = os.getenv("GLUCOTWIN_MODEL", "gemini-2.5-flash")
DEFAULT_EMBED_MODEL = os.getenv("GLUCOTWIN_EMBED_MODEL", "gemini-embedding-001")
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised when the endpoint returns an error that retries cannot fix."""


@dataclass
class Response:
    """Structured response from one LLM call."""

    content: str = ""
    reasoning: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    metadata: dict | None = None


def _post(url: str, body: dict, api_key: str, retries: int = 3, timeout: int = 120) -> dict:
    """POST JSON with exponential backoff on rate limits and server errors."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code in RETRY_STATUS and attempt < retries:
                wait = 2 ** (attempt + 1)
                log.warning("HTTP %s, retrying in %ss: %s", e.code, wait, detail)
                time.sleep(wait)
                continue
            raise LLMError(f"HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            if attempt < retries:
                time.sleep(2 ** (attempt + 1))
                continue
            raise LLMError(f"Network error: {e.reason}") from e


class LLM:
    """Minimal chat-completions client (no SDK)."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = GEMINI_BASE_URL,
        temperature: float | None = 0.3,
        reasoning_effort: str | None = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_mode: bool = False,
    ) -> Response:
        """Generate a response from the LLM given a list of messages."""
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort

        data = _post(f"{self.base_url}/chat/completions", body, self.api_key)

        if not data.get("choices"):
            raise LLMError(f"Empty response: {json.dumps(data)[:300]}")
        message = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return Response(
            content=message.get("content") or "",
            reasoning=message.get("reasoning") or message.get("reasoning_content"),
            # Keep the raw tool_call dicts: they must be echoed back verbatim
            # (Gemini attaches thought signatures to them).
            tool_calls=message.get("tool_calls") or [],
            metadata={
                "model": data.get("model", self.model),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
        )


class EmbeddingModel:
    """Text -> vector via the OpenAI-compatible /embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = GEMINI_BASE_URL,
        batch_size: int = 50,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching to stay under request limits."""
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = _post(
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch},
                self.api_key,
                retries=5,
            )
            ordered = sorted(data["data"], key=lambda d: d.get("index", 0))
            vectors.extend(d["embedding"] for d in ordered)
        return vectors

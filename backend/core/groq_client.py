import json
import logging
from typing import List, Optional

import httpx

from backend.config import settings
from backend.core.observability import trace_span

logger = logging.getLogger(__name__)


def _post_chat_completion(model: str, messages: List[dict], json_mode: bool = False) -> Optional[dict]:
    """
    Low-level Groq chat-completions call, traced as a Langfuse generation span
    (a no-op when Langfuse isn't configured). Returns the raw parsed response
    JSON, or None on any failure (missing key, network, auth) -- callers must
    treat None as "unavailable" and degrade gracefully. This must never be the
    reason a legitimate request fails; RBAC/ABAC filtering at the vector layer
    is the actual security boundary and does not depend on this being up.
    """
    if not settings.GROQ_API_KEY:
        return None

    body = {"model": model, "messages": messages, "temperature": 0}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    with trace_span(f"groq:{model}", as_type="generation", model=model, input=messages) as gen:
        try:
            resp = httpx.post(
                f"{settings.GROQ_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.GROQ_API_KEY}"},
                json=body,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            gen.update(
                output=content,
                usage_details={
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                }
            )
            return data
        except Exception:
            logger.warning("Groq call failed; degrading to unverified.", exc_info=True)
            gen.update(output=None, level="ERROR", status_message="groq_call_failed")
            return None


def call_groq_json(model: str, system_prompt: str, user_prompt: str) -> Optional[dict]:
    """Calls Groq in JSON mode with a system+user prompt pair and parses the JSON object response."""
    data = _post_chat_completion(
        model,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        json_mode=True
    )
    if data is None:
        return None
    try:
        return json.loads(data["choices"][0]["message"]["content"])
    except Exception:
        logger.warning("Failed to parse Groq JSON response.", exc_info=True)
        return None


def call_groq_raw(model: str, user_prompt: str) -> Optional[str]:
    """Calls Groq with a single user message and returns the raw text content (no JSON parsing)."""
    data = _post_chat_completion(model, [{"role": "user", "content": user_prompt}])
    if data is None:
        return None
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return None

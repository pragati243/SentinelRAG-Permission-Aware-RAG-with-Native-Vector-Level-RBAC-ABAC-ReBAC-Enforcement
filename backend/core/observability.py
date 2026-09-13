from contextlib import contextmanager

from langfuse import get_client

from backend.config import settings  # noqa: F401 -- ensures load_dotenv() has run

_client = None


def _get_langfuse():
    """
    Lazily initializes a single cached Langfuse client for the process. Safe to
    call even when LANGFUSE_PUBLIC_KEY/SECRET_KEY are unset -- the SDK returns a
    disabled client that no-ops on every call instead of raising, so call sites
    below never need an "is tracing enabled" branch.
    """
    global _client
    if _client is None:
        _client = get_client()
    return _client


@contextmanager
def trace_span(name: str, as_type: str = "span", **kwargs):
    """
    Thin convenience wrapper around Langfuse's start_as_current_observation.
    No-ops safely when Langfuse isn't configured; nests automatically under
    whatever observation is currently active on the call stack.
    """
    client = _get_langfuse()
    with client.start_as_current_observation(as_type=as_type, name=name, **kwargs) as obs:
        yield obs


def flush():
    """Flush buffered trace events -- call at the end of short-lived scripts/tests."""
    if _client is not None:
        _client.flush()

"""Langfuse tracing if configured, otherwise a no-op decorator (tests and local runs need no account)."""
import os


def observe(*args, **kwargs):
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import observe as lf_observe
        return lf_observe(*args, **kwargs)
    return lambda f: f

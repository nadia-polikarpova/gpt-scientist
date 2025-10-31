"""Utility functions for gpt_scientist."""

import asyncio
from typing import Awaitable, TypeVar

T = TypeVar("T")


def _in_notebook() -> bool:
    """Check if we are running in a Jupyter notebook."""
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return bool(ip) and "IPKernelApp" in ip.config
    except Exception:
        return False


def run_async(coro: Awaitable[T]) -> T:
    """
    Run an async coroutine, handling different contexts (script, notebook, async context).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop: scripts/CLI
        return asyncio.run(coro)

    # Running loop detected
    if not _in_notebook():
        # Contract: sync wrappers forbidden in non-notebook async contexts
        raise RuntimeError(
            "gpt_scientist sync wrapper was called from an async context; "
            "use the async API directly (e.g., `await analyze_csv_async(...)`)."
        )

    # Notebook path: best-effort patch
    try:
        import nest_asyncio
        nest_asyncio.apply(loop)
        return loop.run_until_complete(coro)
    except Exception as e:
        raise RuntimeError(
            "Failed to run in a notebook event loop (nest_asyncio apply/run). "
            "Use the async API with `await` instead."
        ) from e

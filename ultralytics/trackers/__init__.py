# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

__all__ = "BOTSORT", "BYTETracker", "register_tracker"


def __getattr__(name: str):
    """Lazily import tracker backends so opt-in modules do not pull optional tracking deps at package import time."""
    if name == "BOTSORT":
        from .bot_sort import BOTSORT

        return BOTSORT
    if name == "BYTETracker":
        from .byte_tracker import BYTETracker

        return BYTETracker
    if name == "register_tracker":
        from .track import register_tracker

        return register_tracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

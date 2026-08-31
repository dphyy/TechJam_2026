"""Current public-agent showcase; legacy helper imports remain compatible."""
from __future__ import annotations


def __getattr__(name):
    # Historical tests/callers can reproduce the old renderer explicitly. The
    # default CLI must never instantiate the former neural submission agent.
    from . import legacy_showcase
    try:
        return getattr(legacy_showcase, name)
    except AttributeError:
        raise AttributeError(name) from None


def main() -> None:
    from .submission import main as submission_main
    submission_main()


if __name__ == "__main__":
    main()

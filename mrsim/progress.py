"""Minimal dependency-free progress reporting for long-running loops."""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def track(iterable: Iterable[T], total: int, label: str = "") -> Iterator[T]:
    """Yield from iterable while printing progress to stderr.

    On a TTY the line updates in place; otherwise one line is printed per
    ~10% milestone so batch logs stay readable.
    """
    is_tty = sys.stderr.isatty()
    milestone = max(1, total // 10)
    start = time.monotonic()
    for i, item in enumerate(iterable, start=1):
        if is_tty:
            sys.stderr.write(f"\r{label} {i}/{total}")
            if i == total:
                sys.stderr.write("\n")
            sys.stderr.flush()
        elif i % milestone == 0 or i == total:
            elapsed = time.monotonic() - start
            sys.stderr.write(f"{label} {i}/{total} ({elapsed:.1f}s)\n")
            sys.stderr.flush()
        yield item

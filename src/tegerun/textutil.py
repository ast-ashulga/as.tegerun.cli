"""Small text helpers for output that comes from other people's messages."""

from __future__ import annotations

import re

# C0 controls except \t and \n, DEL, and the C1 range: ESC starts terminal escape sequences
# (window titles, cursor moves, clipboard writes on some terminals), so none of it may reach a TTY.
_UNSAFE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def printable(text: object) -> str:
    """`text` with terminal control characters replaced by U+FFFD; newlines and tabs are kept."""
    return _UNSAFE.sub("�", str(text))

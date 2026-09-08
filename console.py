"""
console.py
==========
Make stdout/stderr safe for the characters this project actually prints.

Every script here uses box-drawing rules, ✓/⚠ marks and emoji in its progress
output. On Windows the console defaults to cp1252, which cannot encode any of
them, so the first such print raised UnicodeEncodeError and killed the run —
comparison.py died before training a single detector. The project is
Windows-first (WMI temperatures, P/E core mapping, pywin32), so this is the
common case, not an edge case.

Call enable_unicode_output() at the top of any entry point that prints.
"""

import sys


def enable_unicode_output() -> None:
    """Switch the standard streams to UTF-8, degrading instead of raising.

    errors='replace' matters as much as the encoding: a terminal that cannot
    render a glyph should show a placeholder, never abort the run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is None:
            continue                      # redirected to something non-text
        try:
            reconfigure(encoding='utf-8', errors='replace')
        except (ValueError, OSError):
            # Already detached, or a stream that refuses reconfiguration.
            pass

"""Command-line entry point for gradpulse.

    python -m gradpulse            # branded welcome banner
    python -m gradpulse --version  # concise version + attribution
    gradpulse                      # same, once installed (console_scripts)

Kept deliberately lightweight: the version is read without importing the (heavy)
package, so ``--version`` is instant.

The banner shows a GRADPULSE wordmark with "Pure State Labs Inc." beneath it,
then the version, tagline, and repository URL. Color is rendered with ``rich``
when available *and* output is an interactive terminal; it degrades gracefully:
  * no ``rich`` installed     -> plain (uncolored) banner
  * piped / redirected output -> plain banner
  * ``NO_COLOR`` set          -> rich auto-disables color
  * non-Unicode stdout        -> ASCII '#' wordmark + ASCII separators
(c) 2026 Pure State Labs Inc.
"""
from __future__ import annotations

import argparse
import sys

REPO_URL = "https://github.com/PureStateLabs/gradpulse"
TAGLINE = "A differentiable, multi-solver-validated, open-system pulse optimizer for predictive superconducting-gate fidelities"
ORG = "Pure State Labs Inc."

# GRADPULSE wordmark (ANSI-Shadow style). Block glyphs, 6 rows tall.
_B = "\u2588"   # full block
_GLYPHS = {
    "G": [" \u2588\u2588\u2588\u2588\u2588\u2588\u2557 ", "\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d ", "\u2588\u2588\u2551  \u2588\u2588\u2588\u2557", "\u2588\u2588\u2551   \u2588\u2588\u2551", "\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d", " \u255a\u2550\u2550\u2550\u2550\u2550\u255d "],
    "R": ["\u2588\u2588\u2588\u2588\u2588\u2588\u2557 ", "\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557", "\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d", "\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557", "\u2588\u2588\u2551  \u2588\u2588\u2551", "\u255a\u2550\u255d  \u255a\u2550\u255d"],
    "A": [" \u2588\u2588\u2588\u2588\u2588\u2557 ", "\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557", "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551", "\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2551", "\u2588\u2588\u2551  \u2588\u2588\u2551", "\u255a\u2550\u255d  \u255a\u2550\u255d"],
    "D": ["\u2588\u2588\u2588\u2588\u2588\u2557 ", "\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557", "\u2588\u2588\u2551  \u2588\u2588\u2551", "\u2588\u2588\u2551  \u2588\u2588\u2551", "\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d", "\u255a\u2550\u2550\u2550\u2550\u2550\u255d "],
    "P": ["\u2588\u2588\u2588\u2588\u2588\u2588\u2557 ", "\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557", "\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d", "\u2588\u2588\u2554\u2550\u2550\u2550\u255d ", "\u2588\u2588\u2551     ", "\u255a\u2550\u255d     "],
    "U": ["\u2588\u2588\u2557   \u2588\u2588\u2557", "\u2588\u2588\u2551   \u2588\u2588\u2551", "\u2588\u2588\u2551   \u2588\u2588\u2551", "\u2588\u2588\u2551   \u2588\u2588\u2551", "\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d", " \u255a\u2550\u2550\u2550\u2550\u2550\u255d "],
    "L": ["\u2588\u2588\u2557     ", "\u2588\u2588\u2551     ", "\u2588\u2588\u2551     ", "\u2588\u2588\u2551     ", "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557", "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d"],
    "S": ["\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557", "\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d", "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557", "\u255a\u2550\u2550\u2550\u2550\u2588\u2588\u2551", "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2551", "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d"],
    "E": ["\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557", "\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d", "\u2588\u2588\u2588\u2588\u2588\u2557  ", "\u2588\u2588\u2554\u2550\u2550\u255d  ", "\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557", "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d"],
}

# Box-drawing -> '#' map for the ASCII fallback wordmark.
_ASCII_MAP = {ord(c): "#" for c in "\u2588\u2550\u2551\u2557\u2554\u255d\u255a"}


def _wordmark_lines(unicode: bool) -> list[str]:
    rows = [""] * 6
    for ch in "GRADPULSE":
        glyph = _GLYPHS[ch]
        for i in range(6):
            rows[i] += glyph[i] + " "
    if not unicode:
        rows = [r.translate(_ASCII_MAP) for r in rows]
    return [r.rstrip() for r in rows]


def _read_version() -> str:
    """Resolve the version cheaply, without importing torch/the full package.

    Prefer the co-located ``__init__.py`` -- it always reflects the *running* code,
    whereas installed dist metadata can lag a source checkout (e.g. an editable
    install made at an earlier version). Fall back to importlib.metadata, then a
    sentinel.
    """
    try:
        import pathlib
        import re
        init = pathlib.Path(__file__).with_name("__init__.py").read_text(encoding="utf-8")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("gradpulse")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    return "unknown"


def _encodable(s: str) -> bool:
    """True if stdout can encode ``s`` under its current encoding."""
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        s.encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _unicode_ok() -> bool:
    """True if stdout can encode the fancy banner; otherwise fall back to ASCII."""
    return _encodable("\u2588\u2550\u2551\u00b7")


def _render_rich(ver: str, *, unicode: bool) -> bool:
    """Render the branded banner with rich. Returns False if rich is unavailable
    or the output stream is not an interactive terminal."""
    try:
        from rich.console import Console
        from rich.text import Text
    except Exception:
        return False

    console = Console()
    if not console.is_terminal:
        return False

    lines = _wordmark_lines(unicode)
    width = max(len(s) for s in lines)
    dot = "\u00b7" if unicode else "-"

    art = Text("\n".join(lines), style="bold bright_cyan")
    org = Text(ORG.center(width), style="grey70")
    ver_line = Text()
    ver_line.append("gradpulse ", style="bold white")
    ver_line.append(f"v{ver}", style="bold green")
    tag = Text()
    tag.append(TAGLINE, style="italic grey74")
    tag.append(f"  {dot}  ", style="grey50")
    tag.append("Pure State Labs", style="white")
    url = Text(REPO_URL, style="underline blue")

    console.print()
    console.print(art)
    console.print(org)
    console.print()
    console.print(ver_line)
    console.print(tag)
    console.print(url)
    console.print()
    return True


def banner(ver: str, *, unicode: bool = True) -> str:
    """Plain-text banner. Used when rich is unavailable or output is non-interactive."""
    lines = _wordmark_lines(unicode)
    width = max(len(s) for s in lines)
    dot = "\u00b7" if unicode else "-"
    art = "\n".join(lines)
    return (
        f"\n{art}\n"
        f"{ORG.center(width)}\n\n"
        f"gradpulse v{ver}\n"
        f"{TAGLINE}  {dot}  Pure State Labs\n"
        f"{REPO_URL}\n"
    )


def main(argv=None) -> int:
    ver = _read_version()
    unicode = _unicode_ok()
    parser = argparse.ArgumentParser(
        prog="gradpulse",
        description="Gate-pulse optimization for superconducting qubits, by Pure "
                    "State Labs Inc. Run with no arguments for the welcome banner.",
        epilog=f"Repository: {REPO_URL}",
    )
    sep = "\u00b7" if unicode else "-"
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"gradpulse {ver}\n{ORG} {sep} {REPO_URL}",
    )
    parser.parse_args(argv)

    # No subcommand/flag -> the welcome banner. Try rich; fall back to plain.
    if not _render_rich(ver, unicode=unicode):
        print(banner(ver, unicode=unicode))
    return 0


if __name__ == "__main__":
    sys.exit(main())

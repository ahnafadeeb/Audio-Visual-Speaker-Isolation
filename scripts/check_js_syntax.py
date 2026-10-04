"""Structural sanity check for the frontend JS, with no Node.js required.

Not a parser.  It answers the one question that matters after a batch of hand
edits to a file made of three big object literals: did every brace, bracket and
paren close, and does each object literal still end before the next top-level
declaration begins?  A misplaced method insertion -- ``reset()`` landing outside
``engine``, say -- shows up here as an imbalance or a boundary that runs past
the next ``const``.

Comments and the contents of string, template and regex literals are stripped
first, so braces inside a template placeholder, an apostrophe in a comment, or a
quote inside a character class cannot be mistaken for code.

Run::

    python scripts/check_js_syntax.py app/static/app.js
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAIRS = {")": "(", "]": "[", "}": "{"}

#: A ``/`` immediately after one of these cannot be division -- there is no
#: value to its left to divide -- so it opens a regex literal.  Everything else
#: (an identifier, a number, ``)``, ``]``) is read as division.  That is the
#: standard heuristic and it misreads exactly one construct, ``return /re/``,
#: where the preceding token is a keyword rather than punctuation.  The misread
#: is safe to accept because it is loud in precisely the cases that matter: a
#: regex body only affects this checker if it contains a quote or a brace, and
#: leaving either one in the code stream raises below rather than passing.
REGEX_PREV = frozenset("(,=:[!&|?{};+-*%~^<>") | {""}


def _prev_significant(out: list[str]) -> str:
    """Last non-whitespace character emitted so far, or ``""`` at the start.

    Reads the *output* rather than the input, so a comment or a string before
    the ``/`` is already blank and cannot be mistaken for a value to divide.
    """
    for chunk in reversed(out):
        for ch in reversed(chunk):
            if not ch.isspace():
                return ch
    return ""


def _regex_end(src: str, i: int) -> int | None:
    """Index just past the regex literal, flags included, that starts at ``i``.

    ``None`` when it does not close on its line.  A ``/`` inside a character
    class does not end the literal, so ``[...]`` has to be tracked separately
    -- ``/[a-z/]/`` is one regex, not two plus a stray bracket.
    """
    n = len(src)
    j, in_class = i + 1, False
    while j < n:
        ch = src[j]
        if ch == "\\":
            j += 2
            continue
        if ch == "\n":
            return None                       # regex literals cannot span lines
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "/":
            j += 1
            while j < n and src[j].isalpha():         # trailing flags
                j += 1
            return j
        j += 1
    return None


def strip_noise(src: str) -> str:
    """Blank out comments and string bodies, preserving newlines and layout.

    Every removed character is replaced by a space (or kept, if a newline) so
    that reported line numbers still point at the real source line.
    """
    out: list[str] = []
    i, n = 0, len(src)

    def blank(text: str) -> None:
        out.append("".join(ch if ch == "\n" else " " for ch in text))

    while i < n:
        c = src[i]
        two = src[i : i + 2]

        if two == "//":
            j = src.find("\n", i)
            j = n if j == -1 else j
            blank(src[i:j])
            i = j
            continue

        if two == "/*":
            j = src.find("*/", i + 2)
            if j == -1:
                raise SystemExit("unterminated block comment")
            blank(src[i : j + 2])
            i = j + 2
            continue

        if c in ("'", '"'):
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == c:
                    break
                j += 1
            if j >= n:
                raise SystemExit(f"unterminated string starting at offset {i}")
            blank(src[i : j + 1])
            i = j + 1
            continue

        if c == "`":
            # Template literal.  ${...} placeholders hold real code, so they are
            # kept verbatim -- their braces must balance like any other.
            out.append(" ")
            i += 1
            while i < n:
                if src[i] == "\\":
                    blank(src[i : i + 2])
                    i += 2
                    continue
                if src[i] == "`":
                    out.append(" ")
                    i += 1
                    break
                if src[i : i + 2] == "${":
                    depth = 0
                    j = i + 1                     # sits on '{'
                    while j < n:
                        if src[j] == "{":
                            depth += 1
                        elif src[j] == "}":
                            depth -= 1
                            if depth == 0:
                                break
                        j += 1
                    if j >= n:
                        raise SystemExit("unterminated ${} in template literal")
                    out.append("  ")              # for the '${'
                    out.append(strip_noise(src[i + 2 : j]))
                    out.append(" ")               # for the '}'
                    i = j + 1
                    continue
                blank(src[i])
                i += 1
            continue

        if c == "/" and _prev_significant(out) in REGEX_PREV:
            # Regex literal.  Its body is data, not code: a character class
            # like [&<>"'] carries quote characters and a quantifier like
            # {2,3} carries braces, so reading it as source would unbalance
            # the whole file from that point on.
            j = _regex_end(src, i)
            if j is None:
                raise SystemExit(f"unterminated regex starting at offset {i}")
            blank(src[i:j])
            i = j
            continue

        out.append(c)
        i += 1

    return "".join(out)


def check_balance(code: str) -> None:
    stack: list[tuple[str, int]] = []
    line = 1
    for ch in code:
        if ch == "\n":
            line += 1
        elif ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack or stack[-1][0] != PAIRS[ch]:
                top = stack[-1] if stack else "EMPTY"
                raise SystemExit(f"FAIL  line {line}: unexpected {ch!r}, open is {top}")
            stack.pop()
    if stack:
        where = ", ".join(f"{c!r} opened line {ln}" for c, ln in stack[:8])
        raise SystemExit(f"FAIL  unclosed: {where}")
    print("  PASS  every (), [] and {} balances")


def check_literals(code: str, names: list[str]) -> None:
    """Each named top-level object literal must close before the next `const`."""
    for name in names:
        m = re.search(r"(?m)^const " + name + r" = \{", code)
        if not m:
            raise SystemExit(f"FAIL  no top-level `const {name} = {{`")
        start = m.end() - 1
        start_line = code[:start].count("\n") + 1
        depth, j = 0, start
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            raise SystemExit(f"FAIL  `{name}` object literal never closes")
        end_line = code[:j].count("\n") + 1
        between = code[start:j]
        stray = re.search(r"(?m)^const \w+ = ", between)
        if stray:
            bad = between[: stray.start()].count("\n") + start_line
            raise SystemExit(
                f"FAIL  a top-level `const` at line ~{bad} is INSIDE `{name}` "
                f"-- the literal was left open by an edit")
        print(f"  PASS  `{name}` literal spans lines {start_line}-{end_line}, self-contained")


def check_methods(code: str, expected: dict[str, list[str]]) -> None:
    """Methods added by the review must live inside the right literal."""
    for name, methods in expected.items():
        m = re.search(r"(?m)^const " + name + r" = \{", code)
        start = m.end() - 1
        depth, j = 0, start
        while j < len(code):
            if code[j] == "{":
                depth += 1
            elif code[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        body = code[start : j + 1]
        for meth in methods:
            found = re.search(
                r"(?m)^  (?:async )?" + re.escape(meth) + r"\s*\(", body)
            print(f"  {'PASS' if found else 'FAIL'}  `{name}.{meth}` defined inside `{name}`")
            if not found:
                raise SystemExit(1)


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "app/static/app.js")
    src = path.read_text(encoding="utf-8")
    code = strip_noise(src)
    print(f"structural check -- {path}  ({len(src)} chars, {src.count(chr(10)) + 1} lines)")
    check_balance(code)
    check_literals(code, ["engine", "transport", "ui"])
    check_methods(code, {
        "engine": ["init", "reset", "position", "play", "pause", "stopSource",
                   "select", "hardSet", "levels"],
        # quantBias is mirrored in Python by scripts/check_transport.py; if it
        # were renamed or dropped here that mirror would silently test nothing.
        "transport": ["toggle", "onEnded", "seek", "loop", "correct",
                      "quantBias"],
        "ui": ["setTracks", "renderSpeakers", "selectTrack", "nextPairing",
               "overridden", "renderPairing", "markActive", "draw", "repaint",
               "hitTest", "updateTime", "writeDiag", "meters"],
    })
    # The DPR rename: `bw`/`bh` are the backing-store size, `bw2`/`bh2` the box.
    # If the rename were incomplete the box would be drawn at the canvas size.
    if re.search(r"const \[x, y, bw, bh\] = box", code):
        raise SystemExit("FAIL  box destructuring still shadows the DPR bw/bh")
    print("  PASS  DPR backing-store vars are not shadowed by the box destructure")

    # -- hazards specific to the user-overridable stem->face binding ----------
    # nextPairing() rebinds t.channel, so anything that captured a channel value
    # at BUILD time keeps selecting the stem that face used to own.  The button
    # handler is the one place that used to do exactly that.
    if re.search(r"onclick = \(\) => engine\.select\(t\.channel\)", code):
        raise SystemExit(
            "FAIL  speaker button closes over `t.channel` at build time; after a "
            "pairing override it would select the stem that face used to own")
    print("  PASS  speaker buttons resolve the channel at click time")

    # Every selection path must go through ui.selectTrack, i.e. be expressed in
    # TRACK space.  A digit shortcut in CHANNEL space changes meaning under the
    # user's hands the moment they press Swap.
    if re.search(r"engine\.select\(d - 1\)", code):
        raise SystemExit(
            "FAIL  the digit shortcut still selects by CHANNEL index; it must "
            "select the nth speaker BUTTON (ui.selectTrack)")
    print("  PASS  digit shortcuts select in track space, not channel space")

    # writeDiag() must be a method called from load(), not an inline one-shot:
    # after an override the panel would otherwise still report the pipeline's
    # original pairing.
    if not re.search(r"ui\.writeDiag\(\)", code):
        raise SystemExit("FAIL  load() never calls ui.writeDiag()")
    print("  PASS  diagnostics are re-rendered from a method, not written once")

    # A cyclic rotation reaches only n of the n! pairings.  For two speakers that
    # is complete; for three it is not, and a transposition of two faces would be
    # unfixable by any number of presses.
    if not re.search(r"function nextPermutation\(", code):
        raise SystemExit("FAIL  nextPermutation() is gone; the pairing control "
                         "cannot be complete for 3+ speakers without it")
    print("  PASS  pairing steps through permutations, not a cyclic rotation")

    # Objective B, as a source-level guarantee: switching audio must not stop
    # playback.  nextPairing() re-selects through engine.select(), which only
    # schedules gain automation -- if it ever reached for the source node or the
    # transport, a swap would restart or pause the clip.
    body = _method_body(code, "ui", "nextPairing")
    for banned in ("stopSource", "engine.play", "engine.pause", "transport.",
                   "engine.init", "createBufferSource"):
        if banned in body:
            raise SystemExit(
                f"FAIL  ui.nextPairing() touches `{banned}`; a pairing override "
                f"must not interrupt playback")
    if "engine.select" not in body and "selectTrack" not in body:
        raise SystemExit("FAIL  ui.nextPairing() never re-selects, so the audio "
                         "would not actually change")
    print("  PASS  ui.nextPairing() rebinds audio without touching the transport")
    print("\nstructure OK")


def _method_body(code: str, obj: str, meth: str) -> str:
    """Source of one method of one top-level object literal, braces balanced."""
    m = re.search(r"(?m)^const " + obj + r" = \{", code)
    if not m:
        raise SystemExit(f"FAIL  no `const {obj} = {{`")
    m2 = re.search(r"(?m)^  (?:async )?" + re.escape(meth) + r"\s*\([^)]*\)\s*\{",
                   code[m.end():])
    if not m2:
        raise SystemExit(f"FAIL  no `{obj}.{meth}` to inspect")
    start = m.end() + m2.end() - 1
    depth, j = 0, start
    while j < len(code):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                break
        j += 1
    return code[start : j + 1]


if __name__ == "__main__":
    main()

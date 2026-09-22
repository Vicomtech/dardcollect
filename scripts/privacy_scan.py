#!/usr/bin/env python3
"""Privacy scan for `scripts/validate_harness.py` (advisory check).

Flags machine-identifying paths in the *published* set of files so the repo
stays safe to make public. Extracted from `validate_harness.py` on 2026-09-22
when that script crossed the 600-line god-file cap; the check is unchanged in
behavior, only relocated.

Three holes were closed on 2026-09-22 after real leaks reached the public
repository (incident recorded in `docs/HARNESS_RULES.md`):

  (a) PATTERN hole - only the Windows `Users` home form was matched, so a
      drive-absolute path under any other root passed silently, in prose and in
      code alike.
  (b) SCOPE hole - only README.md, AGENTS.md and docs/*.md were scanned, so a
      real user name living in tests/ was never inspected.
  (c) ENCODING hole - the scan decoded every text file as UTF-8, so a tracked
      UTF-16 log carrying drive-absolute paths matched nothing.
      `_decode_privacy_text` now decodes UTF-16 and reports any file whose
      bytes stay unreadable rather than passing it silently.

The scan covers every tracked text file (`git ls-files` is the authority for
"published") and recognises any drive-absolute literal. Most drive-absolute
strings in a public repository are legitimate (documented example roots,
platform-invariant vendor install directories, synthetic fixtures), so each
allowance is an explicit rule instead of one unreadable alternation.

The separator class is assembled from a character code so this file's own
source cannot match the patterns it defines (a self-reference would flag the
detector on every run). Comments here avoid literal drive paths for the same
reason. The scan's own test fixtures
(`tests/test_validate_harness.py`) assemble their drive paths from parts too.

Binary formats are out of scope by design; Office/PNG author metadata is a
separate, documented residual gap.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_BS = chr(92)
# one separator, backslash or forward slash
_SEP = "[" + _BS * 2 + "/]"

# Synthetic user names used in fixtures and documentation: a home-directory path
# carrying one of these identifies nobody.
_PRIVACY_FAKE_USERS = frozenset(
    {"testuser", "test", "user", "username", "you", "me", "example", "name", "alice", "bob"}
)

# Home-directory paths, any drive letter, with the user name captured so a
# synthetic one can be allowlisted.
_HOME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"[A-Za-z]:" + _SEP + r"Users" + _SEP + r"([^\\/\s\"'`)\]]+)"),
        "Windows home-directory path",
    ),
    (re.compile(r"/(?:home|Users)/([a-z0-9_\-]+)"), "Unix/macOS home-directory path"),
]

# Any drive-absolute literal: a letter, a colon, a separator - not preceded by
# scheme characters (so `https://` is not a match).
_DRIVE_ABS = re.compile(r"(?<![A-Za-z0-9+.\-])[A-Za-z]:" + _SEP)
_URL_LINE = re.compile(r"https?://")

# Allowlisted drive-absolute shapes. Matched against the remainder of the line
# starting at the drive letter, with separators unified to `/`. The rules are:
# platform-invariant vendor install directories, the documented example root,
# a `...` placeholder, a bare drive root with a single segment (a fixture token
# such as `C:/First`), and a home path whose user name is synthetic.
_PRIVACY_ALLOW = re.compile(
    "|".join(
        (
            r"[a-z]:/program files",  # Windows install dir
            r"[a-z]:/tensorrt-",  # vendor install dir
            r"[a-z]:/(?:nvidia|python\d)",  # vendor install dirs
            r"[a-z]:/data(?=/|$|\W)",  # documented example root
            r"[a-z]:/\.\.\.",  # placeholder form
            r"[a-z]:/[^/\s\"'`)\]}]+(?=[\"'`\s,;:}\)\]}]|$)",  # bare root fixture
            r"[a-z]:/users/(?:" + "|".join(sorted(_PRIVACY_FAKE_USERS)) + r")(?=/|$|\W)",
        )
    ),
    re.I,
)

# Text files whose content is committed and therefore published. A text-suffixed
# file that is NOT text (UTF-16 without a BOM, or raw bytes) cannot be scanned,
# so it is surfaced rather than skipped - see `_decode_privacy_text`.
_PRIVACY_TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".cfg",
    ".ini",
    ".mmd",
    ".tex",
}
# Local-only state is never uploaded: scanning it serves no publication purpose.
# Vendored third-party code is not authored here. Matched as path prefixes so a
# directory named e.g. `harness` elsewhere is not silently excluded.
_PRIVACY_EXCLUDE_PREFIXES = (".kilo/worktrees/", "tools/harness/")
_PRIVACY_EXCLUDE_PARTS = {".git", "node_modules", "__pycache__", ".venv"}


def _normalise_seps(text: str) -> str:
    """Collapse separator runs so source and prose spellings compare equal.

    A Python source literal spells a separator twice, prose spells it once; both
    forms must normalise to the same single-separator shape before the allowance
    rules are consulted, or every path written inside source code would look
    like a distinct (and unrecognised) location.
    """
    return re.sub("[" + _BS * 2 + "/]+", "/", text)


def _decode_privacy_text(raw: bytes) -> tuple[str | None, str | None]:
    """Decode a tracked text file, or report why it is unscannable.

    Returns ``(text, None)`` when the bytes decode, or ``(None, reason)`` when
    the file's bytes cannot be examined. The 2026-09-22 residual gap was a
    UTF-16LE `*.txt` whose drive-absolute paths were invisible to a UTF-8 read;
    decoding UTF-16 (BOM or NUL-evidence) closes the encoding hole, and any
    remaining undecodable bytes are REPORTED (not skipped) so an unscannable
    text file becomes a visible warning instead of a silent pass.
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16"), None
        except UnicodeDecodeError:
            return None, "UTF-16 with an undecodable code unit"
    if b"\x00" in raw:
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                text = raw.decode(enc)
            except UnicodeDecodeError:
                continue
            if "\x00" not in text:
                return text, None
        return None, "NUL bytes remain after UTF-16 decoding (binary content?)"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "not valid UTF-8 and not UTF-16"


def _privacy_home_offenders(line: str) -> list[str]:
    """Home-directory pattern labels on a line, skipping synthetic user names."""
    labels: list[str] = []
    for pattern, label in _HOME_PATTERNS:
        for m in pattern.finditer(line):
            if m.group(1).lower() in _PRIVACY_FAKE_USERS:
                continue
            labels.append(label)
    return labels


def _privacy_drive_offenders(line: str) -> list[str]:
    """Drive-absolute literals on a line that no allowance rule covers."""
    offenders: list[str] = []
    if _URL_LINE.search(line):
        return offenders
    pos = 0
    while True:
        m = _DRIVE_ABS.search(line, pos)
        if m is None:
            break
        tail = _normalise_seps(line[m.start() :])
        if _PRIVACY_ALLOW.match(tail):
            pos = m.end()
            continue
        candidate = re.match(r"[A-Za-z]:/[^\"'`\s\)\]},;]*", tail)
        offenders.append(candidate.group(0) if candidate else tail[:40])
        pos = m.end()
    return offenders


def _privacy_candidates(repo_root: Path) -> list[Path]:
    """Text files that would be published, so the scan's scope is the real one.

    `git ls-files` is the authority: it is exactly the published set, so
    gitignored local state (`MEMORY.md`, `.kilo/_metrics/`, generated viewer
    indexes) is not scanned - it is never uploaded, and flagging the machine the
    harness runs on would make the gate noise. When git is unavailable (a plain
    export, or the hermetic test fixture) the scan falls back to a directory
    walk so it still runs rather than silently doing nothing.
    """
    rels: list[str] = []
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        rels = [r for r in out.split() if r]
    except (OSError, subprocess.CalledProcessError):
        rels = []
    if not rels:
        rels = [
            p.relative_to(repo_root).as_posix() for p in sorted(repo_root.rglob("*")) if p.is_file()
        ]
    files: list[Path] = []
    for rel in rels:
        rel_posix = rel.replace(_BS, "/")
        if rel_posix.startswith(_PRIVACY_EXCLUDE_PREFIXES):
            continue
        if any(part in _PRIVACY_EXCLUDE_PARTS for part in rel_posix.split("/")):
            continue
        if Path(rel).suffix.lower() not in _PRIVACY_TEXT_SUFFIXES:
            continue
        files.append(repo_root / rel)
    return files


def scan(repo_root: Path) -> list[str]:
    """Warning-level: personal-data patterns in committed text files.

    Machine-local paths identify the person; the repository is public. Each hit
    is reported for user review (the agent never auto-redacts).
    """
    warnings: list[str] = []
    for f in _privacy_candidates(repo_root):
        if not f.exists():
            continue
        text, reason = _decode_privacy_text(f.read_bytes())
        if text is None:
            warnings.append(
                f"unscannable text file {f.relative_to(repo_root)} -> {reason}; "
                f"the privacy scan cannot read it, so a leak inside would be "
                f"invisible. Convert it to UTF-8 (and delete it from the "
                f"published set if it is a generated artifact)"
            )
            continue
        seen: set[tuple[str, int]] = set()
        for lineno, line in enumerate(text.splitlines(), 1):
            for label in _privacy_home_offenders(line):
                if (label, lineno) in seen:
                    continue
                seen.add((label, lineno))
                warnings.append(
                    f"personal-data pattern ({label}) in "
                    f"{f.relative_to(repo_root)}:{lineno} -> review the hit with "
                    f"the user; redact surgically if it identifies the person "
                    f"(committed history keeps the old bytes)"
                )
            for literal in _privacy_drive_offenders(line):
                if ("drive-absolute path", lineno) in seen:
                    continue
                seen.add(("drive-absolute path", lineno))
                warnings.append(
                    f"personal-data pattern (drive-absolute path) in "
                    f"{f.relative_to(repo_root)}:{lineno} -> `{literal}` is a "
                    f"drive-absolute path that is not a known example or install "
                    f"location; use a placeholder (repo-relative path, tempfile) "
                    f"if it identifies the machine"
                )
    return warnings

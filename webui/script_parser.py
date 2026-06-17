#!/usr/bin/env python3
"""Parse RECORDING-SCRIPT.md into structured, displayable prompts.

Extends the parser in scripts/capture.py with section metadata (title, target
minutes), prompt "kind" (read / spell / spontaneous / conversational / take),
and external-corpus expansion (Harvard sec01, CMU sec02) read from data/raw/.

Pure stdlib. No audio deps — safe to import from the web server.

Public API:
  parse_script(root) -> list[dict]   # one dict per section, each with .prompts
"""
from __future__ import annotations

import re
from pathlib import Path

# Bounded inputs: a hand-authored markdown script, never adversarial.
MAX_LINES = 20_000          # P10 rule 2: hard loop bound
MAX_PROMPT_CHARS = 2_000    # one scripted line is short; guard against runaway

SECTION_RE = re.compile(r"^## Section (\d+):\s+(.+?)(?:\s+—\s*~?(\d+))?\s*(?:min)?\s*$")
SUBSECTION_RE = re.compile(r"^### (\d+)([a-z])\.\s*(.*)$")
ITEM_RE = re.compile(r"^(\d+)\.\s+(.+?)$")
SPEAKER_RE = re.compile(r"^SPEAKER:\s+(.+?)$")
TAKE_RE = re.compile(r"^Take \d+.*:$")
TARGET_MIN_RE = re.compile(r"—\s*~?(\d+)\s*min")


def slug(text: str, n: int = 40) -> str:
    """URL/filename-safe slug. Mirrors scripts/capture.py for path parity."""
    assert isinstance(text, str), "slug needs str"
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "line"


def _kind_for(section_num: int, sub_letter: str) -> str:
    """Classify a prompt by where it sits in the script."""
    if section_num == 10:
        return "spontaneous"   # talk freely, don't read verbatim
    if section_num == 9:
        return "conversational"
    if section_num == 6 and sub_letter == "c":
        return "spell"
    if section_num == 7:
        return "take"
    return "read"


def _section_target_min(header_line: str) -> float:
    m = TARGET_MIN_RE.search(header_line)
    return float(m.group(1)) if m else 0.0


def _new_section(num: int, title: str, target_min: float) -> dict:
    return {
        "section": f"sec{num:02d}",
        "num": num,
        "title": title.strip(),
        "target_min": target_min,
        "prompts": [],
    }


def _append_prompt(section: dict, sub_letter: str, idx: int, text: str) -> None:
    text = text.strip()
    if not text or len(text) > MAX_PROMPT_CHARS:
        return
    sub = sub_letter or ""
    pid = f"{section['section']}{sub}_{idx:03d}"
    section["prompts"].append({
        "id": pid,
        "section": section["section"],
        "sub": sub,
        "idx": idx,
        "text": text,
        "kind": _kind_for(section["num"], sub_letter),
        "slug": slug(text),
    })


def _expand_corpus(root: Path, section: dict, fname: str, label: str) -> bool:
    """Populate sec01/sec02 from a fetched corpus file. Returns True if used."""
    f = root / "data" / "raw" / fname
    if not f.exists():
        return False
    lines = [ln.strip() for ln in f.read_text(errors="ignore").splitlines()]
    lines = [ln for ln in lines if ln][:500]   # P10: bounded
    for i, ln in enumerate(lines, start=1):
        # CMU lines look like: ( arctic_a0001 "the text here" )
        m = re.search(r'"(.+?)"', ln)
        _append_prompt(section, "", i, m.group(1) if m else ln)
    return bool(section["prompts"])


def parse_script(root: Path) -> list[dict]:
    """Return the script as a list of section dicts, each carrying its prompts."""
    root = Path(root)
    path = root / "RECORDING-SCRIPT.md"
    assert path.exists(), f"missing recording script: {path}"

    sections: list[dict] = []
    cur: dict | None = None
    sub_letter = ""
    in_code = False

    for n, raw in enumerate(path.read_text().splitlines()):
        if n > MAX_LINES:
            break
        line = raw.rstrip()

        sm = SECTION_RE.match(line)
        if sm:
            num = int(sm.group(1))
            cur = _new_section(num, sm.group(2), _section_target_min(line))
            sections.append(cur)
            sub_letter = ""
            in_code = False
            continue
        if cur is None:
            continue

        ssm = SUBSECTION_RE.match(line)
        if ssm:
            sub_letter = ssm.group(2)
            continue
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            continue

        speaker = SPEAKER_RE.match(line.strip())
        if speaker:
            _append_prompt(cur, sub_letter, len(cur["prompts"]) + 1, speaker.group(1))
            continue
        im = ITEM_RE.match(line.strip())
        if im:
            _append_prompt(cur, sub_letter, int(im.group(1)), im.group(2))

    _post_expand(root, sections)
    return sections


def _post_expand(root: Path, sections: list[dict]) -> None:
    """Fill external-corpus sections from fetched files when the body was empty."""
    for s in sections:
        if s["num"] == 1 and not s["prompts"]:
            _expand_corpus(root, s, "harvard_1-10.txt", "Harvard")
        elif s["num"] == 2 and not s["prompts"]:
            _expand_corpus(root, s, "cmu_arctic_400.txt", "CMU ARCTIC")
    _parse_special(root, sections)


# Comma-list lines that are plain names (sec06 a/b) and multi-line "Take" blocks
# (sec07) aren't numbered, so the main loop skips them. Recover them here.
def _parse_special(root: Path, sections: list[dict]) -> None:
    path = root / "RECORDING-SCRIPT.md"
    by_num = {s["num"]: s for s in sections}
    num = sub = None
    in_code = False
    take_buf: list[str] = []

    def flush_take(sec: dict) -> None:
        if take_buf:
            txt = " ".join(t.strip() for t in take_buf if t.strip())
            _append_prompt(sec, "", len(sec["prompts"]) + 1, txt)
            take_buf.clear()

    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        sm = SECTION_RE.match(line)
        if sm:
            num, sub, in_code = int(sm.group(1)), None, False
            continue
        ssm = SUBSECTION_RE.match(line)
        if ssm:
            sub = ssm.group(2)
            continue
        if line.lstrip().startswith("```"):
            if num == 7 and in_code:
                flush_take(by_num[7])
            in_code = not in_code
            continue
        if not in_code:
            continue
        if num == 6 and sub in ("a", "b") and "," in line:
            for name in line.split(","):
                name = name.strip().rstrip(".")
                if name:
                    sec = by_num[6]
                    _append_prompt(sec, sub, len(sec["prompts"]) + 1, name)
        elif num == 7:
            if TAKE_RE.match(line.strip()):
                flush_take(by_num[7])
            take_buf.append(line)
    if num == 7:
        flush_take(by_num[7])


if __name__ == "__main__":
    import sys
    secs = parse_script(Path(__file__).resolve().parent.parent)
    total = 0
    for s in secs:
        print(f"{s['section']}  {s['title'][:48]:50s} {len(s['prompts']):4d} prompts  ~{s['target_min']:.0f}min")
        total += len(s["prompts"])
    print(f"\n{len(secs)} sections, {total} prompts")
    sys.exit(0)

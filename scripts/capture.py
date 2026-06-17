#!/usr/bin/env python3
"""Interactive recording helper for Alex's voice corpus.

Parses RECORDING-SCRIPT.md, presents each scripted line one at a time, records
until you press Enter, and saves to data/raw/{section}_{idx:03d}_{slug}.wav.

Resumes automatically by skipping any (section, idx) that already has a file.

Controls per take:
  Enter    start recording  → Enter    stop recording
  then     Enter=keep   r=redo   s=skip   q=quit (resumable)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import sounddevice as sd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_FILE = ROOT / "RECORDING-SCRIPT.md"
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_RATE = 48_000  # capture at 48k, downsample in preprocess

SECTION_RE = re.compile(r"^## Section (\d+):\s+(.+?)(?:\s+—|\s+\(|$)")
SUBSECTION_RE = re.compile(r"^### (\d+)([a-z])\.")
ITEM_RE = re.compile(r"^(\d+)\.\s+(.+?)$")


def parse_script(path: Path) -> Iterator[tuple[str, int, str]]:
    """Yield (section_slug, line_idx, text) for every scripted, numbered line."""
    section: str | None = None
    in_code = False
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        m = SECTION_RE.match(line)
        if m:
            section = f"sec{int(m.group(1)):02d}"
            continue
        sm = SUBSECTION_RE.match(line)
        if sm:
            section = f"sec{int(sm.group(1)):02d}{sm.group(2)}"
            continue
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code or not section:
            continue
        im = ITEM_RE.match(line.strip())
        if im:
            yield section, int(im.group(1)), im.group(2).rstrip()


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "line"


def already_recorded(section: str, idx: int) -> bool:
    return any(RAW_DIR.glob(f"{section}_{idx:03d}_*.wav"))


def record_take(prompt: str) -> np.ndarray:
    print(f"\n  >>> {prompt}")
    input("      [Enter to start] ")
    frames: list[np.ndarray] = []

    def cb(indata, _f, _t, _s):
        frames.append(indata.copy())

    print("      ● recording  ", end="", flush=True)
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", callback=cb):
        input("(Enter to stop) ")
    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1), "int16")
    print(f"      ({len(audio) / SAMPLE_RATE:.1f}s captured)")
    return audio


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", help="Only record lines whose section starts with this (e.g. sec03 or sec05a)")
    ap.add_argument("--list", action="store_true", help="List all lines + recorded status, then exit")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not SCRIPT_FILE.exists():
        print(f"missing: {SCRIPT_FILE}", file=sys.stderr)
        return 1

    items = list(parse_script(SCRIPT_FILE))
    if args.section:
        items = [t for t in items if t[0].startswith(args.section)]
    if not items:
        print("no items matched", file=sys.stderr)
        return 1

    if args.list:
        for sec, idx, text in items:
            mark = "✓" if already_recorded(sec, idx) else " "
            print(f"  [{mark}] {sec}_{idx:03d}  {text[:70]}")
        done = sum(1 for s, i, _ in items if already_recorded(s, i))
        print(f"\n{done}/{len(items)} recorded")
        return 0

    try:
        info = sd.query_devices(sd.default.device[0], "input")
        print(f"Recording from: {info['name']}  ({SAMPLE_RATE} Hz, mono)")
    except Exception as e:
        print(f"Audio device error: {e}", file=sys.stderr)
        return 1

    done = sum(1 for s, i, _ in items if already_recorded(s, i))
    print(f"\nSession: {len(items)} lines, {done} already done.")
    print("Per take: Enter=keep  r=redo  s=skip  q=quit\n")

    for sec, idx, text in items:
        if already_recorded(sec, idx):
            continue
        while True:
            audio = record_take(text)
            choice = input("      keep? [enter=yes  r=redo  s=skip  q=quit]: ").strip().lower()
            if choice == "q":
                print("\nProgress saved. Resume by running again.")
                return 0
            if choice == "s":
                break
            if choice == "r":
                continue
            out = RAW_DIR / f"{sec}_{idx:03d}_{slug(text)}.wav"
            sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
            print(f"      ✓ saved {out.name}")
            break

    print("\nAll lines recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

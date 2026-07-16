#!/usr/bin/env python3
"""Interactive recording helper for the canonical Satraj/Piranesi corpus.

Consumes the same structured compiler as the studio, presents each line one at
a time, and saves to data/raw/{canonical_prompt_id}_{slug}.wav.

Resumes automatically by skipping any canonical prompt ID already recorded.

Controls per take:
  Enter    start recording  → Enter    stop recording
  then     Enter=keep   r=redo   s=skip   q=quit (resumable)
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SAMPLE_RATE = 48_000  # capture at 48k, downsample in preprocess

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webui.prompts import compile_prompts  # noqa: E402


def parse_script(_path: Path | None = None) -> Iterator[tuple[str, str, str]]:
    """Yield (section, canonical_prompt_id, text) from the authoritative compiler."""
    for prompt in compile_prompts(ROOT):
        yield prompt["section"], prompt["id"], prompt["text"]


def slug(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return s[:n] or "line"


def already_recorded(prompt_id: str) -> bool:
    return any(RAW_DIR.glob(f"{prompt_id}_*.wav"))


def record_take(prompt: str) -> Any:
    import numpy as np
    import sounddevice as sd

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
    items = list(parse_script())
    if args.section:
        items = [t for t in items if t[0].startswith(args.section)]
    if not items:
        print("no items matched", file=sys.stderr)
        return 1

    if args.list:
        for _section, prompt_id, text in items:
            mark = "✓" if already_recorded(prompt_id) else " "
            print(f"  [{mark}] {prompt_id}  {text[:70]}")
        done = sum(1 for _, prompt_id, _ in items if already_recorded(prompt_id))
        print(f"\n{done}/{len(items)} recorded")
        return 0

    try:
        import sounddevice as sd

        info = sd.query_devices(sd.default.device[0], "input")
        print(f"Recording from: {info['name']}  ({SAMPLE_RATE} Hz, mono)")
    except Exception as e:
        print(f"Audio device error: {e}", file=sys.stderr)
        return 1

    done = sum(1 for _, prompt_id, _ in items if already_recorded(prompt_id))
    print(f"\nSession: {len(items)} lines, {done} already done.")
    print("Per take: Enter=keep  r=redo  s=skip  q=quit\n")

    for _section, prompt_id, text in items:
        if already_recorded(prompt_id):
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
            import soundfile as sf

            out = RAW_DIR / f"{prompt_id}_{slug(text)}.wav"
            sf.write(out, audio, SAMPLE_RATE, subtype="PCM_16")
            print(f"      ✓ saved {out.name}")
            break

    print("\nAll lines recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

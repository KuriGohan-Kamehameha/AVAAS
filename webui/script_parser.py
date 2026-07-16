#!/usr/bin/env python3
"""Server compatibility façade for the structured AVAAS prompt compiler.

The historical public function is retained so existing callers receive the
same section/prompt shape. `prompts/` is authoritative; Markdown is generated.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .prompts import compile_sections
else:  # Preserve `python webui/script_parser.py` for existing operators.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from webui.prompts import compile_sections


def parse_script(root: Path) -> list[dict]:
    """Return the validated corpus in the legacy section-list shape."""
    return compile_sections(Path(root))


if __name__ == "__main__":
    sections = parse_script(Path(__file__).resolve().parent.parent)
    total = 0
    for section in sections:
        print(
            f"{section['section']}  {section['title'][:48]:50s} "
            f"{len(section['prompts']):4d} prompts  ~{section['target_min']:.0f}min"
        )
        total += len(section["prompts"])
    print(f"\n{len(sections)} sections, {total} prompts")

#!/usr/bin/env python3
"""Materialize prompt sources that AVAAS must not redistribute in Git.

The source response and the derived corpus are both content-addressed.  P10:
network input, parsed lists, sentences, and output bytes all have hard bounds.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HARVARD_URL = "https://www.cs.columbia.edu/~hgs/audio/harvard.html"
HARVARD_SOURCE_SHA256 = "85c3fb1e9ec6bcce9128998eb5263ecbf1e51cb12dd9a869ccc6b2f243b176d8"
HARVARD_DERIVED_SHA256 = "eb79c0bfadc4b3988ee7caa343b84691c2ae1ded100f61fdb6966ef5b5bb9c72"
MAX_SOURCE_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_SENTENCE_CHARS = 2_000
HTTP_TIMEOUT_SECONDS = 30
_LIST_RE = re.compile(r"^List\s+(\d+)$", re.IGNORECASE)


class MaterializationError(RuntimeError):
    """A build-only corpus failed its bounded integrity contract."""


class _HarvardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._tag: str | None = None
        self._parts: list[str] = []
        self._current_list: int | None = None
        self.sentences: dict[int, list[str]] = {number: [] for number in range(1, 11)}

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"h2", "li"}:
            self._tag = tag
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._tag is not None and len(self._parts) < 64:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self._tag:
            return
        text = " ".join("".join(self._parts).split())
        if tag == "h2":
            match = _LIST_RE.fullmatch(text)
            self._current_list = int(match.group(1)) if match else None
        elif self._current_list in self.sentences:
            entries = self.sentences[self._current_list]
            if len(entries) >= 10:
                raise MaterializationError(f"list {self._current_list} exceeds sentence bound")
            if not text or len(text) > MAX_SENTENCE_CHARS:
                raise MaterializationError(f"list {self._current_list} has invalid sentence size")
            entries.append(text)
        self._tag = None
        self._parts = []


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_harvard_100(data: bytes) -> list[str]:
    """Extract lists 1..10 from a bounded, pinned Harvard HTML response."""
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_SOURCE_BYTES:
        raise MaterializationError("Harvard source size outside bounds")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MaterializationError("Harvard source is not UTF-8") from exc
    parser = _HarvardParser()
    try:
        parser.feed(text)
        parser.close()
    except (MaterializationError, ValueError) as exc:
        raise MaterializationError(f"invalid Harvard HTML: {exc}") from exc
    lines: list[str] = []
    for number in range(1, 11):
        entries = parser.sentences[number]
        if len(entries) != 10:
            raise MaterializationError(f"Harvard list {number} has {len(entries)} sentences, expected 10")
        lines.extend(entries)
    if len(lines) != 100:
        raise MaterializationError("Harvard corpus count mismatch")
    return lines


def _download() -> bytes:
    request = Request(HARVARD_URL, headers={"User-Agent": "AVAAS-corpus-builder/1"})
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.geturl() != HARVARD_URL:
                raise MaterializationError("Harvard source redirected away from the pinned URL")
            data = response.read(MAX_SOURCE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise MaterializationError(f"cannot fetch Harvard source: {exc}") from exc
    if not 1 <= len(data) <= MAX_SOURCE_BYTES:
        raise MaterializationError("Harvard download size outside bounds")
    actual = _sha256(data)
    if actual != HARVARD_SOURCE_SHA256:
        raise MaterializationError(f"Harvard source hash mismatch: {actual}")
    return data


def _render(lines: list[str]) -> bytes:
    output = ("\n".join(lines) + "\n").encode("utf-8")
    if len(output) > MAX_OUTPUT_BYTES:
        raise MaterializationError("Harvard derived corpus size outside bounds")
    actual = _sha256(output)
    if actual != HARVARD_DERIVED_SHA256:
        raise MaterializationError(f"Harvard derived hash mismatch: {actual}")
    return output


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _check(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            data = handle.read(MAX_OUTPUT_BYTES + 1)
    except OSError as exc:
        raise MaterializationError(f"cannot read materialized Harvard corpus: {path}") from exc
    if not 1 <= len(data) <= MAX_OUTPUT_BYTES or _sha256(data) != HARVARD_DERIVED_SHA256:
        raise MaterializationError("materialized Harvard corpus failed integrity check")
    lines = data.decode("utf-8").splitlines()
    if len(lines) != 100 or any(not line or len(line) > MAX_SENTENCE_CHARS for line in lines):
        raise MaterializationError("materialized Harvard corpus failed shape check")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--fetch", action="store_true")
    operation.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args(argv)
    destination = args.root / "prompts" / "corpora" / "harvard_100.txt"
    try:
        if args.fetch:
            _write_atomic(destination, _render(extract_harvard_100(_download())))
        _check(destination)
    except MaterializationError as exc:
        parser.exit(1, f"materialization failed: {exc}\n")
    print(f"verified {destination} ({HARVARD_DERIVED_SHA256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

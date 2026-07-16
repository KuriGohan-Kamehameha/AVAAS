# AVAAS — Automated Voice Assimilation and Application System

A self-hostable **browser studio for building a clean voice corpus** — record (or
upload) prompts, standardize them with a deterministic DSP pipeline, and track
readiness for fine-tuning a text-to-speech model on your own voice.

Recording happens in the browser (`getUserMedia`), so a microphone and a modern
browser are all you need on the capture side. Everything is processed and stored
on the server you run — **your recordings never leave your machine.**

## Features

- **Record or upload** — capture from the mic (`MediaRecorder`, with browser AGC /
  noise-suppression / echo-cancellation turned *off* so the signal is raw), or drop
  in any audio file. Everything is normalized through `ffmpeg` to 48 kHz mono.
- **Live input meter** — a peak-hold dBFS meter (Web Audio `AnalyserNode`) with a
  target band and too-loud / too-quiet verdicts, so levels are right before you read.
- **Deterministic standardization pipeline** (`numpy` + `scipy` + `ffmpeg`, no heavy
  ML deps): decode → auto-sample dead-air room tone → spectral-subtraction denoise →
  loudness-normalize (LUFS) → trim silence → resample. The raw 48 kHz take is archived
  (reversible); a processed 24 kHz copy is written for training.
- **Optional speech QC** — if [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
  is installed, each take is transcribed and word-error-rate checked against the prompt;
  degrades gracefully (QC skipped) if the model is unavailable.
- **A structured prompt corpus** — a locally generated `RECORDING-SCRIPT.md` presents
  932 prompts across 13 sections, biased toward phone-call distribution: greetings,
  numbers, names, hold phrases, dialogues, plus Harvard / CMU-ARCTIC prose for phonetic
  and prosodic coverage.
- **Progress + readiness gate** — per-section progress, a manifest of every take, and a
  readiness gate that can (optionally) trigger a downstream training job.

## Quickstart

### Docker (recommended)

```sh
docker build -t avaas .
docker run -d --name avaas --restart unless-stopped \
  -p 127.0.0.1:8731:8731 -v avaas-data:/app/data avaas
# open http://localhost:8731
```

> **Microphone note.** `getUserMedia` only works in a *secure context*:
> `http://localhost` is fine, but any other host needs **HTTPS**. To use AVAAS from
> another device, put it behind a TLS reverse proxy (Caddy, nginx, or
> `tailscale serve`) — a plain `http://<host>:8731` URL will silently block the mic.

### Local

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # plus a system ffmpeg
python scripts/materialize_prompt_corpora.py --fetch
python -m webui.prompts --write
uvicorn webui.server:app --host 127.0.0.1 --port 8731
```

## Using the recording script

The authoritative declarations live under `prompts/`. They compile one human
speaker (`satraj`) into a shared `satraj-piranesi` model corpus with paired Satraj and
Piranesi identity lines. `RECORDING-SCRIPT.md` is generated for humans; change the
structured catalog and run `python -m webui.prompts --write` rather than editing the
Markdown. The complete view is ignored by Git because it includes a build-only corpus;
CI materializes it locally, generates it, then uses `python -m webui.prompts --check`
to reject drift.

The Harvard-derived portion is deliberately absent from Git because AVAAS does
not assert redistribution rights for it. Local and container builds download the
exact pinned source, verify its hash, extract lists 1–10 within fixed bounds, and
verify the derived hash. Do not publish a built image containing that material
unless you have separately confirmed that distribution is authorized.

To run the local verification suite from a fresh checkout:

```sh
python scripts/materialize_prompt_corpora.py --fetch
python -m webui.prompts --write
pytest -q
```

## Layout

```
webui/            FastAPI app: server (+ SSE), DSP, QC, corpus manifest, script parser
webui/static/     single-page studio UI (vanilla JS)
scripts/          optional CLI helpers (terminal capture / batch preprocess)
RECORDING-SCRIPT.md   the prompt corpus (template)
Dockerfile        container image (python3.12 + ffmpeg + deps)
```

## Privacy

`data/` (your recordings, transcripts, and manifest) is git-ignored and never
committed. AVAAS does no network calls except an optional one-time model download
for `faster-whisper` QC.

## Part of the Piranesi voice stack

AVAAS is the **voice-assimilation** half. Its sibling is **[wakeword-training](https://github.com/P1R4N351/wakeword-training)** — a Docker-first [openWakeWord](https://github.com/dscripka/openWakeWord) trainer that turns a wake phrase into a `tflite`/`onnx` detector for edge devices.

They connect end to end: **AVAAS assimilates the voice → wakeword-training detects its wake phrase.** A voice cloned in AVAAS can synthesize personalized positive samples of the wake phrase, so the detector is tuned to that specific speaker rather than generic TTS voices.

## License

MIT — see [LICENSE](LICENSE).

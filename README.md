# AVAAS — Automated Voice Assimilation and Application System

A self-hostable **browser studio for building a clean voice corpus** — record (or
upload) prompts, standardize them with a deterministic DSP pipeline, and track
readiness for fine-tuning a text-to-speech model on your own voice.

Recording happens in the browser (`getUserMedia`), so a microphone and a modern
browser are all you need on the capture side. Everything is processed and stored
on the server you run. Recordings are never sent to a third-party service and leave
the studio only through an explicit, checksummed export to your own training worker.

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
- **One voice, two presentations** — Satraj records one natural human corpus. Paired
  first-person Satraj and Piranesi lines enrich that same voice identity; synthesis
  policy selects General North American/Canadian English for Satraj or King's English
  for Piranesi without asking the speaker to perform a foreign accent.
- **Progress + readiness gate** — per-section progress, a manifest of every take, and a
  readiness gate that can (optionally) trigger a downstream training job.

## Quickstart

### Docker (recommended)

```sh
mkdir -p data
# Native Linux only: sudo chown 10001:10001 data && chmod 0700 data
AVAAS_REVISION="$(git rev-parse HEAD)" docker compose up -d --build
curl --fail http://127.0.0.1:8732/readyz
# open http://localhost:8732
```

Compose runs the exact locked dependency set as UID/GID 10001, with a read-only
application filesystem, dropped capabilities, bounded resources, a loopback-only
canary port, and one explicit writable data directory. It refuses to create a missing
bind path. For a durable Linux deployment, provision (for example)
`/srv/avaas-data` as mode `0700`, owned by `10001:10001`, and set
`AVAAS_DATA_DIR=/srv/avaas-data`.

> **Microphone note.** `getUserMedia` only works in a *secure context*:
> `http://localhost` is fine, but any other host needs **HTTPS**. To use AVAAS from
> another device, put it behind a TLS reverse proxy (Caddy, nginx, or
> `tailscale serve`) — a plain `http://<host>:8731` URL will silently block the mic.

### Local

```sh
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install --require-hashes -r requirements-dev.lock
python scripts/materialize_prompt_corpora.py --fetch
python -m webui.prompts --write
uvicorn webui.server:app --host 127.0.0.1 --port 8731
```

This also requires a system `ffmpeg` and `libsndfile`.

## Voice identity, accents, and model releases

Record every ordinary prompt in your comfortable, natural voice. The corpus has one
source human, `speaker_id=satraj`; `satraj` and `piranesi` are presentation aliases,
not separate speakers or separate training silos. The paired lines intentionally let
the model hear the same vocal identity introducing itself both ways.

Accent and delivery are separate from identity. The expressive worker can condition
the shared voice toward the alias policy at synthesis time: Satraj defaults to
`en-CA` / General North American-Canadian and Piranesi to `en-GB` / received-pronunciation
King's English. Do not imitate the British accent while recording the core corpus.
Optional, clearly labeled calibration prompts may capture safe examples such as warm,
authoritative, urgent, whispered, or projected delivery; only controls that pass the
GPU canary are advertised. Additional language packs remain disabled until their
languages and pronunciations are declared explicitly.

The voice does not mutate online after every take. Releases are versioned:

1. A small accepted reference set can produce an explicitly labeled zero-shot preview.
2. Once the readiness gate has a sufficiently broad, clean corpus, AVAAS seals an
   immutable training/adaptation job and builds a candidate.
3. Listening, identity, accent, intelligibility, latency, organ, and VoIP gates decide
   whether that candidate is promoted; the prior version remains available to roll back.
4. Later accepted recordings can seed a new candidate release, so the voice improves in
   controlled generations rather than changing underneath live calls.

The deterministic Piper fallback uses the vocoder portion of the exact
MIT-licensed LJSpeech `medium` checkpoint as a warm start while creating a new
two-presentation acoustic model. The pin is
`rhasspy/piper-checkpoints@95a4b650bd38716c97caf16d07b2a1734859f91a`, path
`en/en_US/ljspeech/medium/lj-med_1000.ckpt`, SHA-256
`dcf2449bdbdaad09256a08dfac211c59f6b36ce8d3f244fd844a9eb1d7384c7c`.
The training gate rejects a different source, use mode, path, digest, or license.

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

## Backup and recovery

Backups use SQLite's online backup API, include every accepted checksummed recording,
exclude service secrets, and publish append-only bundles only after verification:

```sh
docker exec avaas-studio-canary \
  python /app/scripts/backup.py /app /app/data/backups --backup-id before-upgrade
docker exec avaas-studio-canary \
  python /app/scripts/restore_verify.py \
  /app/data/backups/before-upgrade /unused --verify-only
```

A restore refuses an existing destination and stages a fully verified copy before an
atomic publish. Restore to a new path while the studio is stopped, inspect it, and only
then replace the active data directory; never restore over a live corpus.

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
committed. Runtime captures remain on the self-hosted studio unless you explicitly
export them to a training worker. Image preparation fetches the exact hash-pinned
prompt source, and optional `faster-whisper` QC may download its configured model.

## Part of the Piranesi voice stack

AVAAS is the **voice-assimilation** half. Its sibling is **[wakeword-training](https://github.com/P1R4N351/wakeword-training)** — a Docker-first [openWakeWord](https://github.com/dscripka/openWakeWord) trainer that turns a wake phrase into a `tflite`/`onnx` detector for edge devices.

They connect end to end: **AVAAS assimilates the voice → wakeword-training detects its wake phrase.** A voice cloned in AVAAS can synthesize personalized positive samples of the wake phrase, so the detector is tuned to that specific speaker rather than generic TTS voices.

## License

MIT — see [LICENSE](LICENSE).

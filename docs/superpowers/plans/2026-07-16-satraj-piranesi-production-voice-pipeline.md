# Satraj/Piranesi Production Voice Pipeline Implementation Plan

> **Execution:** Follow this plan task by task with test-driven development and a fresh implementer/reviewer pair for each task. Do not promote fixture audio or change the live default voice until a real Satraj corpus and candidate pass every promotion gate.

**Goal:** Deliver, deploy, and verify a production voice pipeline that records Satraj speaking both as Satraj and as Piranesi, trains one standalone Piper ONNX voice, serves both aliases on internode-0, supports the existing voice organs and 24 kHz VoIP path, and exports validated personalized wakeword positives.

**Architecture:** AVAAS owns structured prompts, transactional recordings, QC, readiness, and immutable job/artifact/export contracts. The canonical private Gitea workspace owns a separately licensed Piper worker plus an engine-neutral TTS router and activation/rollback controls. `wakeword-training` owns validation and staging of AVAAS wakeword exports. All boundaries are checksummed, versioned, bounded, and fail closed.

**Technology:** Python 3.12, FastAPI, SQLite, NumPy/SciPy/libsndfile/ffmpeg, Piper 1.4.2 (`d6975e2`), ONNX Runtime CUDA, systemd, Docker/Compose, vanilla browser JavaScript, pytest/unittest.

## Global constraints

- User-facing and new prompt text says **Satraj**, never **Sat**. Historical OS usernames may remain.
- `satraj` and `piranesi` are aliases for one `satraj-piranesi` acoustic model, not two speakers.
- Masters and accepted takes are immutable/content-addressed. Retakes cannot destroy a previously accepted take.
- Personal recordings, model weights, credentials, and speaker embeddings never enter Git.
- AVAAS remains MIT and does not import/link Piper GPL code. Piper code is isolated under the private Gitea component with GPLv3 notices.
- All request bodies, text, decoded samples, jobs, queues, subprocesses, retries, and waits have fixed bounds.
- `GET` readiness endpoints are pure reads. Fixture jobs are permanently `promotable=false`.
- Live changes come from canonical source commits, are deployed side by side, retain Kokoro fallback, and have tested rollback.
- Changed Python files carry explicit P10 bounded-allocation/event-loop rationale where applicable; the release P10 scan has no unexplained findings.

## Task 1: Freeze repository baselines and release branches

**Repositories:**

- AVAAS: `/Users/sat/Piranesi/projects/repos/AVAAS`
- wakeword-training: `/Users/sat/Piranesi/projects/repos/wakeword-training`
- voice organs: `/Users/sat/Piranesi/projects/repos/piranesi-workspace`

- [ ] Record clean status, remotes, current revisions, and baseline commands in `docs/release/2026-07-16/baseline.md`.
- [ ] Keep AVAAS on `codex/satraj-piranesi-piper-pipeline`.
- [ ] Create `codex/avaas-wakeword-import` in wakeword-training.
- [ ] Create `codex/satraj-piranesi-piper-serving` in piranesi-workspace.
- [ ] Run and record:

```sh
python3 -m compileall -q webui scripts
node --check /tmp/avaas-index.js
python3 -m py_compile generate_dataset.py generate_training_samples.py closed_loop_eval.py wakeword_web.py
bash -n trainer.sh docker-train.sh
docker compose config --quiet
python3 -m unittest discover -s infra/kudzu-vox -p 'test_*.py'
```

Expected: all baseline commands pass; any pre-existing test failure is captured before implementation.

## Task 2: Compile deterministic paired Satraj/Piranesi prompts

**AVAAS files:**

- Create: `prompts/identities.json`
- Create: `prompts/catalog.json`
- Materialize at build/runtime only: `prompts/corpora/harvard_100.txt`
- Create: `prompts/corpora/cmu_arctic_400.txt`
- Create: `prompts/provenance.json`
- Create: `scripts/materialize_prompt_corpora.py`
- Create: `tests/test_capture_cli.py`, `tests/test_container_contract.py`, `tests/test_container_smoke.py`
- Modify: `.dockerignore`, `.gitignore`, `Dockerfile`, `scripts/capture.py`
- Create: `webui/prompts.py`
- Create: `tests/test_prompts.py`
- Modify: `webui/script_parser.py`
- Generate locally/container-only (Git-ignored): `RECORDING-SCRIPT.md`

- [ ] Write failing tests for schema/version enforcement, exact source counts, source SHA-256, stable IDs, grouped identity pairs, required name/spelling/introduction/voicemail/phone/wakeword material, and absence of standalone user-facing `Sat`.
- [ ] Run `pytest -q tests/test_prompts.py`; expect collection/import failure for `webui.prompts`.
- [ ] Implement a bounded stdlib compiler that validates `avaas/identities@v1`, corpus provenance/count/hash, duplicate IDs, required sections, and paired variants.
- [ ] Emit compiled records with `id`, `variant_of`, `identity`, `speaker_id=satraj`, `voice_model_id=satraj-piranesi`, text, kind, section, source, and corpus version.
- [ ] Make `script_parser.parse_script()` consume the compiler and preserve its public section shape.
- [ ] Generate Markdown from structured data; refuse hand-edited drift in `python -m webui.prompts --check`.
- [ ] Run tests and compiler check; expected: 100 Harvard + 400 CMU lines, every required identity base has exactly two variants, zero forbidden UI-name hits.
- [ ] Commit: `feat(prompts): add paired Satraj and Piranesi corpus` with `Authored-by: CPCS`.

## Task 3: Replace mutable JSONL state with transactional SQLite

**AVAAS files:**

- Create: `webui/store.py`
- Create: `webui/migrations/0001_initial.sql`
- Create: `webui/migrations/0002_import_legacy.sql`
- Create: `tests/test_store.py`
- Create: `tests/test_migration.py`
- Modify: `webui/corpus.py`
- Modify: `webui/server.py` (register prompt snapshots at startup; tombstone through the store)

- [ ] Write failing tests for WAL/FULL/foreign-key pragmas, ordered idempotent migrations, immutable takes, a single accepted pointer per prompt, acceptance audit history, tombstones, concurrent retakes, crash rollback, checksum validation, JSONL export, SQLite backup/restore, and bounded legacy manifest import.
- [ ] Run `pytest -q tests/test_store.py tests/test_migration.py`; expect missing `webui.store`.
- [ ] Implement explicit transactions and schema tables: `schema_migrations`, `prompts`, `takes`, `derivatives`, `acceptances`, `acceptance_history`, `qc_results`, `training_jobs`, and `artifacts`.
- [ ] Open one connection per operation with busy timeout; never share connections across request threads.
- [ ] Import legacy JSONL only when the database is new; quarantine malformed/path-escaping records and write a deterministic migration report.
- [ ] Keep `corpus.progress()` as a compatibility façade over pure store queries.
- [ ] Prove backup restore by opening the restored database and revalidating all accepted checksums.
- [ ] Commit: `feat(corpus): make takes transactional and immutable`.

## Task 4: Make ingest, DSP derivatives, and QC fail closed

**AVAAS files:**

- Create: `webui/ingest.py`
- Create: `webui/audio_contracts.py`
- Create: `tests/audio_fixtures.py`
- Create: `tests/test_ingest.py`
- Create: `tests/test_processing.py`
- Create: `tests/test_qc.py`
- Modify: `webui/processing.py`
- Modify: `webui/qc_transcribe.py`
- Modify: `webui/server.py` (bounded multipart ingest and room-tone integration)

- [ ] Write failing adversarial tests for chunked upload ceilings, slow upload deadline, malformed codecs/WAV, compressed and decoded oversize, path traversal, NaN/Inf, silence, clipping, extreme duration, wrong rate/channel/subtype, disconnect cleanup, and a simulated crash after each transaction stage.
- [ ] Write golden tests for 48 kHz PCM16 master, 22,050 Hz Piper derivative, 24 kHz serving derivative, and 16 kHz wake derivative; assert sample format, finite samples, duration tolerance, hashes/parameters, and no source overwrite.
- [ ] Run the focused tests; expect missing ingest/audio-contract APIs.
- [ ] Implement bounded streaming to a staging directory with byte/time counters and SHA-256; decode under a subprocess deadline and decoded-sample ceiling.
- [ ] Fsync files/directories, move by content hash, then commit metadata and accepted pointer atomically.
- [ ] Make denoising conditional on measured improvement and store both selection decision and DSP/tool versions.
- [ ] Return `unavailable` for failed ASR model load/inference; readiness must not treat it as a pass. Require stored reasons for soft-QC overrides.
- [ ] Run `pytest -q tests/test_ingest.py tests/test_processing.py tests/test_qc.py`; expected: all green and no staged residue.
- [ ] Commit: `feat(audio): add bounded immutable ingest and derivatives`.

## Task 5: Bind browser capture to immutable prompt versions

**AVAAS files:**

- Create: `webui/capture_tokens.py`
- Create: `tests/test_api_recording.py`
- Create: `tests/browser/recording.spec.mjs`
- Modify: `webui/server.py`
- Modify: `webui/static/index.html`

- [ ] Write failing API tests for short-lived signed capture tokens bound to prompt ID/version, stale/mismatched/replayed tokens, bounded multipart upload, non-destructive failed retake, audited accept/tombstone operations, bounded SSE subscribers, room-tone limits, and typed non-sensitive errors.
- [ ] Write mocked-media browser tests for microphone denial, device loss, start-record/navigate/save race, separate room-tone/clip chunks, upload rejection, retry, playback, retake, and explicit recoverable UI state.
- [ ] Run focused API/browser tests and confirm the capture-token endpoint is absent.
- [ ] Add `POST /api/captures`, require the token on `/api/clip`, stage/process/QC before acceptance, and emit state only after commit.
- [ ] Make `DELETE` create a tombstone rather than unlinking audio. Add an explicit accept-with-override endpoint requiring a reason.
- [ ] Replace shared browser chunk state with per-capture objects; save the token captured at record start, never `FLAT[cur]` at save time.
- [ ] Replace all visible/internal “Sat” forms with “Satraj”; expose both identity variants and their pairing.
- [ ] Run browser syntax and browser suite; expected: navigation cannot relabel audio and failed retake preserves the previous accepted take.
- [ ] Commit: `feat(studio): bind recordings to Satraj and Piranesi prompts`.

## Task 6: Make readiness pure and build immutable Piper job bundles

**AVAAS files:**

- Create: `webui/readiness.py`
- Create: `webui/training_jobs.py`
- Create: `webui/contracts.py`
- Create: `schemas/training-job-v1.schema.json`
- Create: `tests/test_readiness.py`
- Create: `tests/test_training_jobs.py`
- Modify: `webui/trainer.py`
- Modify: `webui/server.py`

- [ ] Write failing tests for mandatory-source gaps, identity-group coverage, duration/section/kind thresholds, checksum/corruption, QC availability/overrides, frozen grouped splits, license ledger, trainer health, and proof that repeated `GET /api/progress` performs zero writes/launches.
- [ ] Write state-machine tests for every legal/illegal transition, compare-and-swap, one active job, idempotency, restart recovery, retry/deadline ceilings, cancellation, and fixture non-promotability.
- [ ] Write bundle rejection tests for absent `READY`, path escape, executable extras, unknown schema, duplicate IDs, invalid split, checksum/license gap, and hyperparameter bounds.
- [ ] Implement pure readiness reports with machine-readable blockers.
- [ ] Replace the Chatterbox launcher with persisted `idle -> queued -> validating -> staging -> running -> evaluating -> packaging -> succeeded` jobs.
- [ ] Produce `avaas/training-job@v1` into a fresh immutable directory, validate it, fsync, then write `READY` last. Include only accepted 22,050 Hz derivatives.
- [ ] Keep auto-train off by default; a bounded scheduler transition may enqueue one job, but no GET route may mutate state.
- [ ] Commit: `feat(training): emit bounded Piper training jobs`.

## Task 7: Validate voice artifacts and export wakeword bundles

**AVAAS files:**

- Create: `schemas/voice-model-v1.schema.json`
- Create: `schemas/wakeword-export-v1.schema.json`
- Create: `webui/artifacts.py`
- Create: `webui/wakeword_export.py`
- Create: `tests/fixtures/voice-model-v1/`
- Create: `tests/fixtures/wakeword-export-v1/`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_wakeword_export.py`
- Modify: `webui/server.py`

- [ ] Write failing tests for unknown schemas, traversal/symlinks, missing/unexpected files, duplicate aliases, checksum/license gaps, incompatible rate/encoding, CPU/CUDA/evaluation failure, and fixture `promotable=false` enforcement.
- [ ] Implement `avaas/voice-model@v1` validation; require aliases `satraj` and `piranesi` to point to the same ONNX hash and require `READY` last.
- [ ] Implement `avaas/wakeword-export@v1` with declared phrase allow-list, artifact/prosody provenance, deterministic 16 kHz PCM16 normalization, checksums, and immutable output.
- [ ] Add authenticated artifact/job status APIs only; do not expose activation or enrollment anonymously.
- [ ] Commit: `feat(contracts): validate voice and wakeword bundles`.

## Task 8: Harden AVAAS packaging, operations, and CI

**AVAAS files:**

- Create: `requirements.lock`
- Create: `requirements-dev.lock`
- Create: `.github/workflows/ci.yml`
- Create: `compose.yaml`
- Create: `scripts/backup.py`
- Create: `scripts/restore_verify.py`
- Modify: `Dockerfile`
- Modify: `entrypoint.sh`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] Write packaging tests/checks for a non-root UID, digest-pinned base, locked hashes, read-only root, explicit writable mounts, health check, request/resource limits, and no secrets/personal data in image or Git.
- [ ] Pin all Python/system/tool versions and preserve license/provenance data in the image.
- [ ] Add backup using SQLite backup API plus checksummed content storage; make restore verification open and validate the restored corpus.
- [ ] CI runs prompt drift, unit/adversarial/migration/backup/browser tests, compile, JS syntax, lint/type, dependency/license/secret/vulnerability scans, and container smoke.
- [ ] Document real-corpus promotion versus nonpromotable fixture validation.
- [ ] Commit: `build: harden AVAAS release and CI`.

## Task 9: Import and actually stage AVAAS positives in wakeword-training

**wakeword-training files:**

- Create: `avaas_import.py`
- Create: `stage_personalized_positives.py`
- Create: `tests/test_avaas_import.py`
- Create: `tests/test_stage_personalized_positives.py`
- Create: `tests/fixtures/avaas-wakeword-export-v1/`
- Create: `.github/workflows/avaas-contract.yml`
- Modify: `generate_dataset.py`
- Modify: `trainer.sh`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`
- Modify: `README.md`
- Modify: `README-docker.md`
- Modify: `RELEASE_CHECKLIST.md`

- [ ] Write failing tests that validate schema, hashes, safe relative paths/no symlinks, mono 16 kHz PCM16, duration/sample ceilings, phrase allow-list, artifact identity, duplicate policy, and append-only destination copies.
- [ ] Write failing staging tests that prove imported files are deterministically split and copied into the run’s upstream `positive_train`/`positive_test` directories after `--generate_clips` and before normalize/augment.
- [ ] Preserve upstream generic Piper positives; enforce separate and total minimum/maximum counts; never let personalized samples replace diversity.
- [ ] Run `python3 -m unittest discover -s tests -v`; expect missing importer/stager first, then green.
- [ ] Invoke the CLI twice against the fixture; the second import is idempotent and no sample is duplicated.
- [ ] Run a trainer dry run and inspect actual upstream run directories, not only the dataset manifest.
- [ ] Commit and push: `feat(dataset): import and stage AVAAS wakeword positives`.

## Task 10: Add immutable Piper validation, activation, and rollback

**piranesi-workspace files:**

- Create: `infra/kudzu-vox/piper/COPYING`
- Create: `infra/kudzu-vox/piper/README.md`
- Create: `infra/kudzu-vox/piper/requirements.lock`
- Create: `infra/kudzu-vox/piper/validate_bundle.py`
- Create: `infra/kudzu-vox/piper/activate_bundle.py`
- Create: `infra/kudzu-vox/piper/tests/test_bundle.py`
- Create: `infra/kudzu-vox/piper/tests/test_activation.py`

- [ ] Copy the complete GPLv3 license and document the MIT-router/GPL-worker service boundary and base-model license obligations.
- [ ] Write failing tests for all `avaas/voice-model@v1` rejection cases, safe file opening, promotability, alias hash equality, atomic symlink activation, previous-pointer retention, concurrent activation lock, and rollback.
- [ ] Implement validation without executing/loading ONNX. Only a validated immutable bundle can be activated.
- [ ] Make activation and rollback local privileged CLIs with locks; no network activation endpoint.
- [ ] Run focused tests, activate a fixture, roll back, and verify both pointers/checksums.
- [ ] Commit: `feat(tts): add immutable Piper artifact activation`.

## Task 11: Add bounded Piper worker and training worker

**piranesi-workspace files:**

- Create: `infra/kudzu-vox/piper/worker.py`
- Create: `infra/kudzu-vox/piper/train_job.py`
- Create: `infra/kudzu-vox/piper/preflight.py`
- Create: `infra/kudzu-vox/piper/tests/test_worker.py`
- Create: `infra/kudzu-vox/piper/tests/test_train_job.py`
- Create: `infra/kudzu-vox/piper/tests/test_preflight.py`
- Create: `infra/kudzu-vox/kudzu-piper.service.tmpl`
- Create: `infra/kudzu-vox/kudzu-piper-trainer.service.tmpl`

- [ ] Write failing tests for training-job schema/checksums/path/license/split/parameter bounds, `READY`, idempotency, legal persisted state transitions, cancellation/deadlines, resumability, immutable result publication, and permanently nonpromotable fixture results.
- [ ] Write GPU preflight tests for disk, host RAM, VRAM, GPU health/temperature/lane occupancy and fail closed when telemetry is missing.
- [ ] Pin Piper 1.4.2 commit `d6975e2`, Python/PyTorch/CUDA/Lightning/espeak-ng, base checkpoint URL/hash/license, seed, split hash, and bounded profile.
- [ ] Export ONNX/config and run CPU plus CUDA finite/non-silent/shape checks before packaging a result.
- [ ] Write worker tests with a fake Piper session for text/speed/alias bounds, serialization, queue/full/deadline/cancellation, chunk limits, and structured health/readiness.
- [ ] Keep framework allocations/event loops explicitly bounded at application boundaries and record the P10 deviation.
- [ ] Commit: `feat(tts): add pinned Piper trainer and GPU worker`.

## Task 12: Put an engine-neutral TTS router in front of Piper and Kokoro

**piranesi-workspace files:**

- Create: `infra/kudzu-vox/tts-router.py`
- Create: `infra/kudzu-vox/tts-kokoro.py`
- Create: `infra/kudzu-vox/test_tts_router.py`
- Create: `infra/kudzu-vox/test_tts_audio_contract.py`
- Create: `infra/kudzu-vox/test_tts_telephony.py`
- Create: `infra/kudzu-vox/kudzu-tts-router.service.tmpl`
- Create: `infra/kudzu-vox/kudzu-kokoro.service.tmpl`
- Modify: `infra/kudzu-vox/tts-server.py`

- [ ] Preserve current Kokoro behavior as a loopback fallback worker and reconcile its live silence-fallback/error-log drift into canonical source.
- [ ] Write failing contract tests for local `GET|POST /tts`, exact 24 kHz mono PCM16 WAV/Content-Length/headers, existing Kokoro voice routing, identical Satraj/Piranesi Piper hashes, OpenAI POST PCM/WAV, bearer auth, typed errors, text/body/speed/language bounds, queue saturation, deadlines, client disconnect, worker death, and readiness distinct from liveness.
- [ ] Write streaming tests for exact 20 ms/960-byte 24 kHz frames, final partial frame, order, fixed buffers, backpressure, cancellation, and explicit end/error.
- [ ] Write 24-to-8 kHz G.711 mu-law golden tests for 160-byte/20 ms frames, pacing drift, malformed lengths, silence keepalive, slow consumer, disconnect, and barge-in cancellation.
- [ ] Route only `satraj`/`piranesi` to Piper; unknown current IDs route unchanged to Kokoro. Require remote POST auth; permit query GET only on the explicit local listener.
- [ ] Commit: `feat(tts): serve Piper through organ and VoIP contracts`.

## Task 13: Integrate canonical deployment without premature promotion

**piranesi-workspace files:**

- Modify: `infra/kudzu-vox/profiles/TEMPLATE.env`
- Modify: `infra/kudzu-vox/profiles/internode-0.env`
- Modify: `infra/kudzu-vox/differentiate.sh`
- Modify: `infra/kudzu-vox/remote-install.sh`
- Modify: `infra/kudzu-vox/deploy.sh`
- Modify: `infra/kudzu-vox/README.md`
- Create: `infra/kudzu-vox/test_deploy_render.py`
- Create: `infra/kudzu-vox/verify-tts-canary.py`

- [ ] Add genes for router/Piper/Kokoro ports, binds, queue/deadline/text ceilings, artifact roots, active pointer, GPU device, and secret-file path. Validate every gene before staging.
- [ ] Render/install distinct units and locked environments, non-root users, read-only roots, explicit writable paths, resource limits, rotating logs, and health checks.
- [ ] Keep `VOX_VOICE=am_michael` in the production profile until a real promotable artifact exists. Add a separate canary alias/port instead of lying about promotion.
- [ ] Make `verify-tts-canary.py` exercise legacy, OpenAI PCM/WAV, both aliases, Kokoro fallback, concurrency, disconnect, worker failure/fallback, and rollback.
- [ ] Test rendered units/config in a temporary staging root and prove the rollback command restores the prior unit/port/model pointer.
- [ ] Commit and push: `deploy(tts): add canary Piper router on internode-0`.

## Task 14: Build a bounded nonpromotable fixture end to end

- [ ] Generate a tiny deterministic, explicitly synthetic/sacrificial AVAAS corpus fixture; mark all derived jobs/results/artifacts `promotable=false`.
- [ ] Run AVAAS prompt -> accepted fixture derivatives -> frozen splits -> `avaas/training-job@v1`.
- [ ] On internode-0 RTX 3090, pass resource preflight and run the bounded Piper fine-tune/export smoke profile without disturbing live inference.
- [ ] Validate ONNX/config on CPU and CUDA, package `avaas/voice-model@v1`, activate only on the canary pointer, and verify alias equality.
- [ ] Exercise `/tts`, `/v1/audio/speech` PCM/WAV, streaming frames, VoIP mu-law conversion, worker death/fallback, and rollback with decoded audio inspection.
- [ ] Export `Hey Piranesi` to `avaas/wakeword-export@v1`, import it into wakeword-training, and prove files enter actual upstream training directories.
- [ ] Record commands, revisions, hashes, timing, RAM/VRAM, and failures in `docs/release/2026-07-16/fixture-e2e.md`.

Expected: the entire mechanical pipeline passes; the fixture remains impossible to select as the live personal voice.

## Task 15: Side-by-side deploy the recording studio

- [ ] Inventory all existing AVAAS data copies and take a fresh consistent SQLite/files snapshot before migration.
- [ ] Build AVAAS from the pushed feature commit, sign/record the image digest, and migrate only a copied dataset.
- [ ] Deploy to a side-by-side port/container, run health, prompt count/hash, empty/new-user progress, migration, recording, failed-retake, backup, and restore tests.
- [ ] Exercise the real hosted HTTPS UI with microphone permission and verify every visible identity uses Satraj/Piranesi correctly.
- [ ] Switch the Tailscale HTTPS front only after the canary passes; retain the previous container/image/data snapshot and execute one rollback rehearsal.
- [ ] Record exact deployed commit/image/snapshot and post-deploy checks in `docs/release/2026-07-16/studio-deploy.md`.

## Task 16: P10, security, review, soak, and canonical push gates

- [ ] Run all three complete test suites plus direct CLI/service/browser invocations.
- [ ] Run format/lint/type/dependency/license/secret/vulnerability scans and the NASA P10 scanner; explain only the two approved bounded runtime deviations.
- [ ] Run an independent specification and code-quality review for each task, then a final cross-repository review.
- [ ] On the canary, capture cold/warm p50/p95, real-time factor, GPU/RAM/VRAM, concurrent organ/phone load, disconnect, worker death, Kokoro fallback, and a 30-minute soak.
- [ ] Execute artifact pointer rollback, router rollback, and studio rollback, then restore the canary and reverify.
- [ ] Push all feature commits to their actual canonical GitHub/Gitea remotes and record exact remote commit IDs.
- [ ] Do not set `VOX_VOICE=piranesi`, call the fixture personal, or claim the final model exists. After Satraj records the readiness corpus, repeat production training/evaluation and only then promote the real model and change the canonical profile.

## Release acceptance

The release is accepted when the hosted studio addresses Satraj correctly, presents paired Piranesi prompts, preserves recordings transactionally, emits valid Piper/wakeword contracts, the GPU canary passes organ/OpenAI/VoIP/rollback tests, the wakeword importer actually stages personalized positives, P10/security/CI evidence is green, and every deployed artifact is traceable to pushed canonical commits. The personal Piranesi voice is promoted only after real recordings and production evaluation pass.

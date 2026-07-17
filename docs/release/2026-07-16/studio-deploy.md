# AVAAS studio deployment evidence — 2026-07-16

## Release identity

- Canonical branch: `codex/satraj-piranesi-piper-pipeline`
- Hardening commit: `dbfb9396a2f06689b5c7d3b7d5e18805c4a85171`
- Deployed regression-fix commit: `4ac8cf646faaf531512130da9c7a1cdb3167c3aa`
- Commit trailer on both commits: `Authored-by: CPCS`
- Image ID: `sha256:0630e1566c0811c94e427a3cdc7a280211bdaca46808b642ad9d8c1ddeffc360`
- OCI revision label: `4ac8cf646faaf531512130da9c7a1cdb3167c3aa`
- Runtime: UID/GID `10001:10001`, read-only root, all capabilities dropped,
  `no-new-privileges`, bounded CPU/RAM/PIDs/logs, and one explicit writable bind.

The image was built on branch-0 from a clean detached checkout of the exact
canonical GitHub commit. The deployed container is `voice-canary`, bound only to
`127.0.0.1:8732`; the prior `voice` container remains available on
`127.0.0.1:8731` for immediate rollback.

## Data preservation

- Pre-deploy source: `/srv/avaas-data`
- Pre-deploy archive:
  `/srv/avaas-backups/2026-07-16-before-dbfb9396a2f0/avaas-data.tar`
- Archive SHA-256:
  `bdc31f50ca75f5825c7987b41769f3a30f4e3dfb9ad00299b13273a6a5a753dc`
- Active pristine data: `/srv/avaas-data-canary-4ac8cf646faa`
- Preserved end-to-end test evidence:
  `/srv/avaas-canary-evidence-4ac8cf646faa`

The active data tree starts with zero takes and zero accepted recordings. Its
QC cache was warmed before promotion. An append-only pristine backup verified as
`avaas/backup@v1` with manifest SHA-256
`60c293039dd3f10b4b29dbccae574e74ffd6ac657422678557b387bad60b59af`.

## Verification

- Python: `126 passed, 1 deselected`
- Browser state: `4 passed`
- Prompt drift: 932 prompts verified
- Ruff, compileall, Bash syntax, and Compose validation: pass
- Runtime dependency audit: no known vulnerabilities
- Changed production source: zero P10 warnings/errors
- Independent important-profile vulnerability scan: zero findings
- Container health/readiness: healthy, four migrations, 932 prompts
- Identity inventory: 796 shared, 68 Satraj, 68 Piranesi
- Standalone user-facing `Sat`: absent
- Browser console warnings/errors on the hosted studio: none

The real Debian container canary exercised:

1. prompt-bound, expiring, one-use capture capability issuance;
2. a valid 48 kHz upload through the deterministic DSP pipeline;
3. contracted master-48k, serve-24k, piper-22050, and wake-16k artifacts;
4. review-required playback and reasoned human acceptance;
5. an immutable backup containing one take and four checked files;
6. verified atomic restore of that bundle;
7. failed-retake preservation of the previously accepted take;
8. capture-token replay rejection; and
9. retake deletion returning recorded progress to zero.

That canary found a platform-specific decoder mismatch: Debian ffmpeg emits
float WAVE_FORMAT_EXTENSIBLE, which libsndfile names `WAVEX`, while macOS ffmpeg
names the same WAVE/pcm_f32le output `WAV`. Commit `4ac8cf6` added a regression
and accepts both container labels while preserving exact codec, channel, rate,
frame, duration, and signal bounds. The complete end-to-end proof then passed.

## Hosted promotion and rollback

The public tailnet URL is:

`https://piranesi-branch-0.tail8d99b6.ts.net/`

Tailscale Serve port 443 now proxies `http://127.0.0.1:8732`. The unrelated
port-18789 handler was preserved. Promotion to 8732, rollback to 8731, and final
promotion to 8732 were all exercised with live HTTP checks. A guarded rollback is:

```sh
sudo tailscale serve --bg --yes http://127.0.0.1:8731
```

Re-promotion is:

```sh
sudo tailscale serve --bg --yes http://127.0.0.1:8732
```

The HTTPS page sends no-store caching, a same-origin microphone permissions
policy, CSP, no-referrer, nosniff, frame denial, and same-origin connection/media
restrictions.

## Recording state

The studio is ready for Satraj to record. Ordinary prompts must be read in his
comfortable natural accent. Satraj and Piranesi lines feed the same human voice
identity. The rendered Piranesi prompt explicitly reports recording metadata
`en-CA / natural / neutral` and synthesis target `piranesi.en-gb.neutral`; it does
not ask Satraj to perform King's English. Hardware microphone permission remains
a user-controlled browser action and was not granted during automated testing.

This deployment makes the capture corpus production-ready. It does not claim
that a personal voice model exists before Satraj records an accepted reference
and the subsequent model candidate passes listening, identity, accent, organ,
VoIP, latency, and rollback gates.

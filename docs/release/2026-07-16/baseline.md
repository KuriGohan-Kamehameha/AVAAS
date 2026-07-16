# Voice Pipeline Repository Baseline

- Captured: `2026-07-16T17:26:09-04:00` (`America/Toronto`)
- Release plan: `docs/superpowers/plans/2026-07-16-satraj-piranesi-production-voice-pipeline.md`
- Author: CPCS
- Pushes performed: none

## Repository revisions

### AVAAS

- Path: `/Users/sat/Piranesi/projects/repos/AVAAS`
- Branch: `codex/satraj-piranesi-piper-pipeline`
- Upstream: `origin/codex/satraj-piranesi-piper-pipeline`
- Commit: `c908a655e290980ebe48a503dd42aca5025bb925`
- Commit subject: `docs: plan Satraj and Piranesi voice release`
- Status before this baseline document: not clean; the pre-existing untracked file
  `.superpowers/sdd/progress.md` was present and was not modified or staged.
- Remotes:
  - `origin` fetch: `https://github.com/KuriGohan-Kamehameha/AVAAS.git`
  - `origin` push: `https://github.com/KuriGohan-Kamehameha/AVAAS.git`

### wakeword-training

- Path: `/Users/sat/Piranesi/projects/repos/wakeword-training`
- Branch created: `codex/avaas-wakeword-import`
- Base branch: `main`
- Upstream: none; the branch has not been pushed.
- Commit: `24148c0124b41ca2aebf01a85d163a1b8fc4aae3`
- Commit subject: `README: tie to AVAAS — part of the Piranesi voice stack`
- Status after branch creation and baseline commands: clean.
- Remotes:
  - `origin` fetch: `https://github.com/KuriGohan-Kamehameha/wakeword-training.git`
  - `origin` push: `https://github.com/KuriGohan-Kamehameha/wakeword-training.git`
  - `upstream` fetch: `https://github.com/fitoori/wakeword-training.git`
  - `upstream` push: `https://github.com/fitoori/wakeword-training.git`

### piranesi-workspace voice organs

- Path: `/Users/sat/Piranesi/projects/repos/piranesi-workspace`
- Branch created: `codex/satraj-piranesi-piper-serving`
- Base branch: `main`
- Upstream: none; the branch has not been pushed.
- Commit: `79c6db9c5f0263d60b601a6341a4191b9cd7efbe`
- Commit subject: `heartbeat piranesi 20260716T210626Z`
- Status after branch creation and baseline commands: clean.
- Remotes:
  - `origin` fetch: `http://100.72.144.19:18792/piranesi-org/workspace.git`
    (partial clone, `blob:none`)
  - `origin` push: `http://100.72.144.19:18792/piranesi-org/workspace.git`

## Baseline commands

### AVAAS

The inline JavaScript was extracted without its HTML tags before syntax checking:

```sh
awk '/^[[:space:]]*<script>[[:space:]]*$/{inside=1; next} /^[[:space:]]*<\/script>[[:space:]]*$/{inside=0} inside{print}' webui/static/index.html > /tmp/avaas-index.js
```

Result: exit `0`; `/tmp/avaas-index.js` contained 326 JavaScript lines.

```sh
python3 -m compileall -q webui scripts
```

Result: exit `0`; no output.

```sh
node --check /tmp/avaas-index.js
```

Result: exit `0`; no output.

### wakeword-training

```sh
python3 -m py_compile generate_dataset.py generate_training_samples.py closed_loop_eval.py wakeword_web.py
```

Result: exit `0`; no output.

```sh
bash -n trainer.sh docker-train.sh
```

Result: exit `0`; no output.

```sh
docker compose config --quiet
```

Result: exit `0`; no output.

### piranesi-workspace voice organs

```sh
python3 -m unittest discover -s infra/kudzu-vox -p 'test_*.py'
```

Result: exit `5`.

```text
----------------------------------------------------------------------
Ran 0 tests in 0.000s

NO TESTS RAN
```

## Pre-existing baseline issues

1. The voice-organ unittest command does not exercise the existing suite. Four
   matching modules are present (`test_threat_voice.py`, `test_voxactions.py`,
   `test_voxarray.py`, and `test_voxd.py`), but they define pytest-style top-level
   test functions rather than `unittest.TestCase` cases. Consequently,
   `unittest discover` finds zero tests and exits `5`.
2. AVAAS started with the untracked task ledger
   `.superpowers/sdd/progress.md`. It is deliberately excluded from this release
   baseline commit.

No production behavior was changed while capturing this baseline.

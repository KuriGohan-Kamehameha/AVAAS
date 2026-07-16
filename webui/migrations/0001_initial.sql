CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    applied_at TEXT NOT NULL
);

CREATE TABLE prompts (
    prompt_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    speaker_id TEXT NOT NULL,
    voice_model_id TEXT NOT NULL,
    identity TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    section TEXT NOT NULL,
    kind TEXT NOT NULL,
    source TEXT NOT NULL,
    snapshot_sha256 TEXT NOT NULL CHECK(length(snapshot_sha256) = 64),
    created_at TEXT NOT NULL,
    PRIMARY KEY (prompt_id, corpus_version)
);

CREATE TABLE takes (
    take_id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    source TEXT NOT NULL,
    raw_sha256 TEXT NOT NULL CHECK(length(raw_sha256) = 64),
    raw_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (take_id, prompt_id, corpus_version),
    FOREIGN KEY (prompt_id, corpus_version)
        REFERENCES prompts(prompt_id, corpus_version)
);

CREATE TABLE derivatives (
    derivative_id TEXT PRIMARY KEY,
    take_id TEXT NOT NULL,
    purpose TEXT NOT NULL,
    sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
    path TEXT NOT NULL,
    sample_rate INTEGER NOT NULL CHECK(sample_rate > 0),
    channels INTEGER NOT NULL CHECK(channels > 0),
    sample_format TEXT NOT NULL,
    frames INTEGER NOT NULL CHECK(frames > 0),
    duration_ms INTEGER NOT NULL CHECK(duration_ms >= 0),
    parameters_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (take_id, purpose, sha256),
    FOREIGN KEY (take_id) REFERENCES takes(take_id)
);

CREATE TABLE qc_results (
    qc_id TEXT PRIMARY KEY,
    take_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    flags_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    transcript TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (take_id) REFERENCES takes(take_id)
);

CREATE TABLE acceptances (
    prompt_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    take_id TEXT,
    generation INTEGER NOT NULL CHECK(generation >= 0),
    accepted_at TEXT,
    override_reason TEXT,
    PRIMARY KEY (prompt_id, corpus_version),
    FOREIGN KEY (prompt_id, corpus_version)
        REFERENCES prompts(prompt_id, corpus_version),
    FOREIGN KEY (take_id, prompt_id, corpus_version)
        REFERENCES takes(take_id, prompt_id, corpus_version)
);

CREATE TABLE acceptance_history (
    event_id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    take_id TEXT,
    action TEXT NOT NULL CHECK(action IN ('accept', 'repoint', 'tombstone')),
    reason TEXT,
    generation INTEGER NOT NULL CHECK(generation > 0),
    created_at TEXT NOT NULL,
    UNIQUE (prompt_id, corpus_version, generation),
    FOREIGN KEY (prompt_id, corpus_version)
        REFERENCES prompts(prompt_id, corpus_version),
    FOREIGN KEY (take_id) REFERENCES takes(take_id)
);

CREATE TABLE take_tombstones (
    take_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (take_id) REFERENCES takes(take_id)
);

CREATE TABLE training_jobs (
    job_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    job_id TEXT,
    schema_name TEXT NOT NULL,
    root_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    promotable INTEGER NOT NULL CHECK(promotable IN (0, 1)),
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES training_jobs(job_id)
);

CREATE INDEX derivatives_take_idx ON derivatives(take_id, purpose);
CREATE INDEX history_prompt_idx ON acceptance_history(prompt_id, corpus_version, generation);
CREATE INDEX takes_prompt_idx ON takes(prompt_id, corpus_version, created_at);

CREATE TRIGGER schema_migrations_no_update BEFORE UPDATE ON schema_migrations
BEGIN SELECT RAISE(ABORT, 'migration receipts are immutable'); END;
CREATE TRIGGER schema_migrations_no_delete BEFORE DELETE ON schema_migrations
BEGIN SELECT RAISE(ABORT, 'migration receipts are immutable'); END;
CREATE TRIGGER prompts_no_update BEFORE UPDATE ON prompts
BEGIN SELECT RAISE(ABORT, 'prompts are immutable'); END;
CREATE TRIGGER prompts_no_delete BEFORE DELETE ON prompts
BEGIN SELECT RAISE(ABORT, 'prompts are immutable'); END;
CREATE TRIGGER takes_no_update BEFORE UPDATE ON takes
BEGIN SELECT RAISE(ABORT, 'takes are immutable'); END;
CREATE TRIGGER takes_no_delete BEFORE DELETE ON takes
BEGIN SELECT RAISE(ABORT, 'takes are immutable'); END;
CREATE TRIGGER derivatives_no_update BEFORE UPDATE ON derivatives
BEGIN SELECT RAISE(ABORT, 'derivatives are immutable'); END;
CREATE TRIGGER derivatives_no_delete BEFORE DELETE ON derivatives
BEGIN SELECT RAISE(ABORT, 'derivatives are immutable'); END;
CREATE TRIGGER qc_results_no_update BEFORE UPDATE ON qc_results
BEGIN SELECT RAISE(ABORT, 'qc results are immutable'); END;
CREATE TRIGGER qc_results_no_delete BEFORE DELETE ON qc_results
BEGIN SELECT RAISE(ABORT, 'qc results are immutable'); END;
CREATE TRIGGER acceptance_history_no_update BEFORE UPDATE ON acceptance_history
BEGIN SELECT RAISE(ABORT, 'acceptance history is immutable'); END;
CREATE TRIGGER acceptance_history_no_delete BEFORE DELETE ON acceptance_history
BEGIN SELECT RAISE(ABORT, 'acceptance history is immutable'); END;
CREATE TRIGGER take_tombstones_no_update BEFORE UPDATE ON take_tombstones
BEGIN SELECT RAISE(ABORT, 'tombstones are immutable'); END;
CREATE TRIGGER take_tombstones_no_delete BEFORE DELETE ON take_tombstones
BEGIN SELECT RAISE(ABORT, 'tombstones are immutable'); END;

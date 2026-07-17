CREATE TABLE voice_training_jobs (
    job_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    engine TEXT NOT NULL CHECK(engine IN ('cosyvoice3', 'piper')),
    profile TEXT NOT NULL,
    promotable INTEGER NOT NULL CHECK(promotable IN (0, 1)),
    fixture INTEGER NOT NULL CHECK(fixture IN (0, 1)),
    manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256) = 64),
    bundle_path TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK(state IN (
        'queued', 'validating', 'staging', 'running', 'evaluating',
        'packaging', 'succeeded', 'failed', 'cancelled'
    )),
    generation INTEGER NOT NULL CHECK(generation >= 0),
    attempts INTEGER NOT NULL CHECK(attempts BETWEEN 0 AND 3),
    deadline_epoch INTEGER NOT NULL CHECK(deadline_epoch > 0),
    active_slot INTEGER NOT NULL DEFAULT 1 CHECK(active_slot = 1),
    detail TEXT NOT NULL,
    created_at_epoch INTEGER NOT NULL CHECK(created_at_epoch >= 0),
    updated_at_epoch INTEGER NOT NULL CHECK(updated_at_epoch >= created_at_epoch),
    CHECK(fixture = 0 OR promotable = 0)
);

CREATE TABLE voice_training_job_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK(generation >= 0),
    detail TEXT NOT NULL,
    event_at_epoch INTEGER NOT NULL CHECK(event_at_epoch >= 0),
    FOREIGN KEY (job_id) REFERENCES voice_training_jobs(job_id),
    UNIQUE (job_id, generation)
);

CREATE UNIQUE INDEX voice_training_jobs_one_active
    ON voice_training_jobs(active_slot)
    WHERE state IN (
        'queued', 'validating', 'staging', 'running', 'evaluating', 'packaging'
    );
CREATE INDEX voice_training_jobs_state_idx
    ON voice_training_jobs(state, updated_at_epoch);
CREATE INDEX voice_training_job_events_job_idx
    ON voice_training_job_events(job_id, generation);

CREATE TRIGGER voice_training_jobs_guard_update BEFORE UPDATE ON voice_training_jobs
WHEN NOT (
    OLD.job_id = NEW.job_id
    AND OLD.idempotency_key = NEW.idempotency_key
    AND OLD.engine = NEW.engine
    AND OLD.profile = NEW.profile
    AND OLD.promotable = NEW.promotable
    AND OLD.fixture = NEW.fixture
    AND OLD.manifest_sha256 = NEW.manifest_sha256
    AND OLD.bundle_path = NEW.bundle_path
    AND OLD.active_slot = NEW.active_slot
    AND NEW.generation = OLD.generation + 1
    AND NEW.updated_at_epoch >= OLD.updated_at_epoch
    AND (
        (
            NEW.attempts = OLD.attempts
            AND NEW.deadline_epoch = OLD.deadline_epoch
            AND (
                (OLD.state = 'queued' AND NEW.state IN ('validating', 'failed', 'cancelled'))
                OR (OLD.state = 'validating' AND NEW.state IN ('staging', 'failed', 'cancelled'))
                OR (OLD.state = 'staging' AND NEW.state IN ('running', 'failed', 'cancelled'))
                OR (OLD.state = 'running' AND NEW.state IN ('evaluating', 'failed', 'cancelled'))
                OR (OLD.state = 'evaluating' AND NEW.state IN ('packaging', 'failed', 'cancelled'))
                OR (OLD.state = 'packaging' AND NEW.state IN ('succeeded', 'failed', 'cancelled'))
            )
        )
        OR (
            OLD.state = 'failed'
            AND NEW.state = 'queued'
            AND NEW.attempts = OLD.attempts + 1
            AND NEW.attempts <= 3
            AND NEW.deadline_epoch > OLD.deadline_epoch
        )
    )
)
BEGIN SELECT RAISE(ABORT, 'illegal voice training job transition'); END;

CREATE TRIGGER voice_training_jobs_no_delete BEFORE DELETE ON voice_training_jobs
BEGIN SELECT RAISE(ABORT, 'voice training jobs are immutable'); END;
CREATE TRIGGER voice_training_job_events_no_update BEFORE UPDATE ON voice_training_job_events
BEGIN SELECT RAISE(ABORT, 'voice training job events are immutable'); END;
CREATE TRIGGER voice_training_job_events_no_delete BEFORE DELETE ON voice_training_job_events
BEGIN SELECT RAISE(ABORT, 'voice training job events are immutable'); END;

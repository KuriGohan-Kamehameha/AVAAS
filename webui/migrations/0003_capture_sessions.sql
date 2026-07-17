CREATE TABLE capture_sessions (
    nonce TEXT PRIMARY KEY CHECK(length(nonce) = 64),
    token_sha256 TEXT NOT NULL UNIQUE CHECK(length(token_sha256) = 64),
    prompt_id TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    generation INTEGER NOT NULL CHECK(generation >= 0),
    issued_at INTEGER NOT NULL CHECK(issued_at >= 0),
    expires_at INTEGER NOT NULL CHECK(expires_at > issued_at),
    state TEXT NOT NULL CHECK(state IN ('issued', 'consumed')),
    consumed_at INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prompt_id, corpus_version)
        REFERENCES prompts(prompt_id, corpus_version)
);

CREATE TABLE capture_session_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    nonce TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('issued', 'consumed')),
    event_at TEXT NOT NULL,
    FOREIGN KEY (nonce) REFERENCES capture_sessions(nonce)
);

CREATE INDEX capture_sessions_prompt_idx
    ON capture_sessions(prompt_id, corpus_version, generation, state);
CREATE INDEX capture_session_events_nonce_idx
    ON capture_session_events(nonce, event_id);

CREATE TRIGGER capture_sessions_guard_update BEFORE UPDATE ON capture_sessions
WHEN NOT (
    OLD.state = 'issued' AND NEW.state = 'consumed'
    AND OLD.consumed_at IS NULL AND NEW.consumed_at IS NOT NULL
    AND OLD.nonce = NEW.nonce
    AND OLD.token_sha256 = NEW.token_sha256
    AND OLD.prompt_id = NEW.prompt_id
    AND OLD.corpus_version = NEW.corpus_version
    AND OLD.generation = NEW.generation
    AND OLD.issued_at = NEW.issued_at
    AND OLD.expires_at = NEW.expires_at
    AND OLD.created_at = NEW.created_at
)
BEGIN SELECT RAISE(ABORT, 'capture sessions permit only one transition'); END;

CREATE TRIGGER capture_sessions_no_delete BEFORE DELETE ON capture_sessions
BEGIN SELECT RAISE(ABORT, 'capture sessions are immutable receipts'); END;
CREATE TRIGGER capture_session_events_no_update BEFORE UPDATE ON capture_session_events
BEGIN SELECT RAISE(ABORT, 'capture session events are immutable'); END;
CREATE TRIGGER capture_session_events_no_delete BEFORE DELETE ON capture_session_events
BEGIN SELECT RAISE(ABORT, 'capture session events are immutable'); END;

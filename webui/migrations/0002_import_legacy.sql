CREATE TABLE legacy_imports (
    manifest_sha256 TEXT PRIMARY KEY CHECK(length(manifest_sha256) = 64),
    manifest_path TEXT NOT NULL,
    report_json TEXT NOT NULL,
    imported_count INTEGER NOT NULL CHECK(imported_count >= 0),
    quarantined_count INTEGER NOT NULL CHECK(quarantined_count >= 0),
    applied_at TEXT NOT NULL
);

CREATE TABLE legacy_quarantine (
    manifest_sha256 TEXT NOT NULL,
    line_number INTEGER NOT NULL CHECK(line_number > 0),
    line_sha256 TEXT NOT NULL CHECK(length(line_sha256) = 64),
    reason TEXT NOT NULL,
    PRIMARY KEY (manifest_sha256, line_number),
    FOREIGN KEY (manifest_sha256) REFERENCES legacy_imports(manifest_sha256)
);

CREATE TRIGGER legacy_imports_no_update BEFORE UPDATE ON legacy_imports
BEGIN SELECT RAISE(ABORT, 'legacy imports are immutable'); END;
CREATE TRIGGER legacy_imports_no_delete BEFORE DELETE ON legacy_imports
BEGIN SELECT RAISE(ABORT, 'legacy imports are immutable'); END;
CREATE TRIGGER legacy_quarantine_no_update BEFORE UPDATE ON legacy_quarantine
BEGIN SELECT RAISE(ABORT, 'legacy quarantine is immutable'); END;
CREATE TRIGGER legacy_quarantine_no_delete BEFORE DELETE ON legacy_quarantine
BEGIN SELECT RAISE(ABORT, 'legacy quarantine is immutable'); END;

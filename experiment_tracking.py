from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def collect_hyperparameters(args: Any) -> dict[str, Any]:
    items: dict[str, Any] = {}
    for name in dir(args):
        if name.startswith("_"):
            continue
        value = getattr(args, name)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            items[name] = value
    return items


@dataclass
class SQLiteExperimentTracker:
    db_path: Path
    script_name: str
    backend: str
    run_id: str
    log_path: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        if not self.enabled:
            self.conn = None
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    @classmethod
    def from_env(
        cls,
        *,
        script_name: str,
        backend: str,
        run_id: str,
        log_path: str | None = None,
    ) -> "SQLiteExperimentTracker":
        enabled = os.environ.get("EXPERIMENT_TRACKING", "1") != "0"
        db_path = Path(os.environ.get("EXPERIMENT_DB_PATH", "./logs/experiments.sqlite3"))
        return cls(db_path=db_path, script_name=script_name, backend=backend, run_id=run_id, log_path=log_path, enabled=enabled)

    def _init_schema(self) -> None:
        assert self.conn is not None
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                script_name TEXT NOT NULL,
                backend TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                finished_at_utc TEXT,
                log_path TEXT,
                notes TEXT
            );

            CREATE TABLE IF NOT EXISTS params (
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                value_type TEXT NOT NULL,
                value_text TEXT,
                value_real REAL,
                value_int INTEGER,
                PRIMARY KEY (run_id, name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                phase TEXT NOT NULL,
                step INTEGER,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at_utc TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                metadata_json TEXT,
                recorded_at_utc TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER,
                name TEXT NOT NULL,
                message TEXT,
                recorded_at_utc TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run_name_step ON metrics(run_id, name, step);
            CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(name);
            CREATE INDEX IF NOT EXISTS idx_params_name ON params(name);
            """
        )
        self.conn.commit()

    def start_run(self, *, notes: str | None = None) -> None:
        if not self.enabled:
            return
        assert self.conn is not None
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs(run_id, script_name, backend, status, started_at_utc, finished_at_utc, log_path, notes)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (self.run_id, self.script_name, self.backend, "running", utc_now_iso(), self.log_path, notes),
        )
        self.conn.commit()

    def _param_row(self, value: Any) -> tuple[str, str | None, float | None, int | None]:
        if isinstance(value, bool):
            return ("bool", str(value), None, int(value))
        if isinstance(value, int):
            return ("int", str(value), None, value)
        if isinstance(value, float):
            return ("float", str(value), value, None)
        if value is None:
            return ("null", None, None, None)
        return ("text", str(value), None, None)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.conn is not None
        rows = []
        for name, value in sorted(params.items()):
            value_type, value_text, value_real, value_int = self._param_row(value)
            rows.append((self.run_id, name, value_type, value_text, value_real, value_int))
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO params(run_id, name, value_type, value_text, value_real, value_int)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def log_metric(self, *, phase: str, name: str, value: float, step: int | None = None) -> None:
        if not self.enabled:
            return
        assert self.conn is not None
        self.conn.execute(
            """
            INSERT INTO metrics(run_id, phase, step, name, value, recorded_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (self.run_id, phase, step, name, float(value), utc_now_iso()),
        )
        self.conn.commit()

    def log_artifact(self, *, name: str, num_bytes: int, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        assert self.conn is not None
        self.conn.execute(
            """
            INSERT INTO artifacts(run_id, name, bytes, metadata_json, recorded_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                name,
                int(num_bytes),
                json.dumps(metadata, sort_keys=True) if metadata is not None else None,
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def log_event(self, *, name: str, message: str | None = None, step: int | None = None) -> None:
        if not self.enabled:
            return
        assert self.conn is not None
        self.conn.execute(
            """
            INSERT INTO events(run_id, step, name, message, recorded_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (self.run_id, step, name, message, utc_now_iso()),
        )
        self.conn.commit()

    def finish(self, *, status: str, notes: str | None = None) -> None:
        if not self.enabled:
            return
        assert self.conn is not None
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, finished_at_utc = ?, notes = COALESCE(?, notes)
            WHERE run_id = ?
            """,
            (status, utc_now_iso(), notes, self.run_id),
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def latest_runs(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH latest_metric AS (
            SELECT
                m.run_id,
                m.phase,
                m.name,
                m.value,
                ROW_NUMBER() OVER (
                    PARTITION BY m.run_id, m.phase, m.name
                    ORDER BY COALESCE(m.step, -1) DESC, m.id DESC
                ) AS rn
            FROM metrics m
            WHERE (m.phase = 'roundtrip' AND m.name IN ('val_bpb', 'val_loss'))
               OR (m.phase = 'val' AND m.name IN ('val_bpb', 'val_loss'))
               OR (m.phase = 'train' AND m.name = 'train_loss')
        )
        SELECT
            r.run_id,
            r.backend,
            r.status,
            r.started_at_utc,
            MAX(CASE WHEN lm.phase = 'roundtrip' AND lm.name = 'val_bpb' AND lm.rn = 1 THEN lm.value END) AS roundtrip_val_bpb,
            MAX(CASE WHEN lm.phase = 'roundtrip' AND lm.name = 'val_loss' AND lm.rn = 1 THEN lm.value END) AS roundtrip_val_loss,
            MAX(CASE WHEN lm.phase = 'val' AND lm.name = 'val_bpb' AND lm.rn = 1 THEN lm.value END) AS val_bpb,
            MAX(CASE WHEN lm.phase = 'val' AND lm.name = 'val_loss' AND lm.rn = 1 THEN lm.value END) AS val_loss,
            MAX(CASE WHEN lm.phase = 'train' AND lm.name = 'train_loss' AND lm.rn = 1 THEN lm.value END) AS train_loss
        FROM runs r
        LEFT JOIN latest_metric lm ON lm.run_id = r.run_id
        GROUP BY r.run_id, r.backend, r.status, r.started_at_utc
        ORDER BY r.started_at_utc DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def runs_for_group(conn: sqlite3.Connection, group: str, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH group_runs AS (
            SELECT run_id
            FROM params
            WHERE name = 'experiment_group' AND value_text = ?
        ),
        latest_metric AS (
            SELECT
                m.run_id,
                m.phase,
                m.name,
                m.value,
                ROW_NUMBER() OVER (
                    PARTITION BY m.run_id, m.phase, m.name
                    ORDER BY COALESCE(m.step, -1) DESC, m.id DESC
                ) AS rn
            FROM metrics m
            JOIN group_runs g ON g.run_id = m.run_id
            WHERE (m.phase = 'roundtrip' AND m.name IN ('val_bpb', 'val_loss'))
               OR (m.phase = 'val' AND m.name IN ('val_bpb', 'val_loss'))
               OR (m.phase = 'train' AND m.name = 'train_loss')
        ),
        labels AS (
            SELECT run_id, value_text AS experiment_label
            FROM params
            WHERE name = 'experiment_label'
        ),
        comments AS (
            SELECT run_id, value_text AS experiment_comment
            FROM params
            WHERE name = 'experiment_comment'
        )
        SELECT
            r.run_id,
            r.backend,
            r.status,
            r.started_at_utc,
            r.notes,
            l.experiment_label,
            c.experiment_comment,
            MAX(CASE WHEN lm.phase = 'roundtrip' AND lm.name = 'val_bpb' AND lm.rn = 1 THEN lm.value END) AS roundtrip_val_bpb,
            MAX(CASE WHEN lm.phase = 'val' AND lm.name = 'val_bpb' AND lm.rn = 1 THEN lm.value END) AS val_bpb,
            MAX(CASE WHEN lm.phase = 'train' AND lm.name = 'train_loss' AND lm.rn = 1 THEN lm.value END) AS train_loss
        FROM runs r
        JOIN group_runs g ON g.run_id = r.run_id
        LEFT JOIN latest_metric lm ON lm.run_id = r.run_id
        LEFT JOIN labels l ON l.run_id = r.run_id
        LEFT JOIN comments c ON c.run_id = r.run_id
        GROUP BY r.run_id, r.backend, r.status, r.started_at_utc, r.notes, l.experiment_label, c.experiment_comment
        ORDER BY r.started_at_utc DESC
        LIMIT ?
        """,
        (group, limit),
    ).fetchall()


def parameter_effect(conn: sqlite3.Connection, name: str, metric: str, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH latest_target AS (
            SELECT
                m.run_id,
                m.phase,
                m.value,
                ROW_NUMBER() OVER (
                    PARTITION BY m.run_id
                    ORDER BY CASE WHEN m.phase = 'roundtrip' THEN 0 ELSE 1 END,
                             COALESCE(m.step, -1) DESC,
                             m.id DESC
                ) AS rn
            FROM metrics m
            WHERE m.name = ?
              AND m.phase IN ('roundtrip', 'val', 'train')
        )
        SELECT
            r.run_id,
            r.backend,
            p.value_text,
            p.value_real,
            p.value_int,
            lt.phase AS metric_phase,
            lt.value AS metric_value,
            r.started_at_utc
        FROM params p
        JOIN runs r ON r.run_id = p.run_id
        LEFT JOIN latest_target lt ON lt.run_id = r.run_id AND lt.rn = 1
        WHERE p.name = ?
        ORDER BY r.started_at_utc DESC
        LIMIT ?
        """,
        (metric, name, limit),
    ).fetchall()


def format_param_value(row: sqlite3.Row) -> str:
    if row["value_real"] is not None:
        return str(row["value_real"])
    if row["value_int"] is not None:
        return str(row["value_int"])
    return row["value_text"] or "NULL"


def print_rows(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Query experiment tracking SQLite data.")
    parser.add_argument(
        "--db",
        default=os.environ.get("EXPERIMENT_DB_PATH", "./logs/experiments.sqlite3"),
        help="Path to the SQLite database.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    latest = subparsers.add_parser("latest", help="Show the latest runs.")
    latest.add_argument("--limit", type=int, default=20)

    group = subparsers.add_parser("group", help="Show runs for one experiment group.")
    group.add_argument("name", help="Experiment group name.")
    group.add_argument("--limit", type=int, default=50)

    param = subparsers.add_parser("param", help="Show one hyperparameter against a metric.")
    param.add_argument("name", help="Hyperparameter name, for example rope_base.")
    param.add_argument("--metric", default="val_bpb", help="Metric name to compare against.")
    param.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()
    conn = connect(Path(args.db))

    if args.command == "latest":
        rows = latest_runs(conn, args.limit)
        print_rows(
            ["run_id", "backend", "status", "train_loss", "val_loss", "val_bpb", "roundtrip_val_bpb", "started_at_utc"],
            [
                [
                    row["run_id"],
                    row["backend"],
                    row["status"],
                    f"{row['train_loss']:.4f}" if row["train_loss"] is not None else "",
                    f"{row['val_loss']:.4f}" if row["val_loss"] is not None else "",
                    f"{row['val_bpb']:.4f}" if row["val_bpb"] is not None else "",
                    f"{row['roundtrip_val_bpb']:.4f}" if row["roundtrip_val_bpb"] is not None else "",
                    row["started_at_utc"],
                ]
                for row in rows
            ],
        )
        return

    if args.command == "group":
        rows = runs_for_group(conn, args.name, args.limit)
        print_rows(
            ["run_id", "label", "backend", "status", "train_loss", "val_bpb", "roundtrip_val_bpb", "comment", "started_at_utc"],
            [
                [
                    row["run_id"],
                    row["experiment_label"] or "",
                    row["backend"],
                    row["status"],
                    f"{row['train_loss']:.4f}" if row["train_loss"] is not None else "",
                    f"{row['val_bpb']:.4f}" if row["val_bpb"] is not None else "",
                    f"{row['roundtrip_val_bpb']:.4f}" if row["roundtrip_val_bpb"] is not None else "",
                    row["experiment_comment"] or row["notes"] or "",
                    row["started_at_utc"],
                ]
                for row in rows
            ],
        )
        return

    rows = parameter_effect(conn, args.name, args.metric, args.limit)
    print_rows(
        ["run_id", "backend", args.name, "metric_phase", args.metric, "started_at_utc"],
        [
            [
                row["run_id"],
                row["backend"],
                format_param_value(row),
                row["metric_phase"] or "",
                f"{row['metric_value']:.4f}" if row["metric_value"] is not None else "",
                row["started_at_utc"],
            ]
            for row in rows
        ],
    )


if __name__ == "__main__":
    main()

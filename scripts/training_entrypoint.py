#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from experiment_tracking import ensure_experiment_schema


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_db_path() -> Path:
    configured = Path(os.environ.get("EXPERIMENT_DB_PATH", "./logs/experiments.sqlite3"))
    return configured if configured.is_absolute() else (ROOT_DIR / configured).resolve()


def ensure_launch_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS launch_requests (
            launch_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            requested_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            status TEXT NOT NULL,
            trainer_mode TEXT NOT NULL,
            resolved_script_name TEXT,
            platform TEXT,
            device_kind TEXT,
            device_count INTEGER,
            pid INTEGER,
            experiment_group TEXT,
            experiment_label TEXT,
            message TEXT,
            config_json TEXT,
            command_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_launch_requests_requested_at ON launch_requests(requested_at_utc DESC);
        CREATE INDEX IF NOT EXISTS idx_launch_requests_run_id ON launch_requests(run_id);
        """
    )
    conn.commit()


def open_launch_db() -> sqlite3.Connection:
    db_path = resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_experiment_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    ensure_launch_schema(conn)
    return conn


def read_json_stdin() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def detect_environment() -> dict[str, Any]:
    system = platform.system()
    info: dict[str, Any] = {
        "platform": system.lower(),
        "platformLabel": system,
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "pythonExecutable": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "repoRoot": str(ROOT_DIR),
        "dbPath": str(resolve_db_path()),
        "cudaAvailable": False,
        "cudaDeviceCount": 0,
        "cudaDeviceNames": [],
        "mlxAvailable": False,
        "supportedTrainerModes": [],
        "defaultNprocPerNode": 1,
        "defaultDeviceKind": "cpu",
        "launchSupport": "unsupported",
        "launchMessage": "No supported training backend detected.",
    }

    try:
        import torch

        info["torchVersion"] = torch.__version__
        info["cudaAvailable"] = bool(torch.cuda.is_available())
        if info["cudaAvailable"]:
            info["cudaDeviceCount"] = int(torch.cuda.device_count())
            info["cudaDeviceNames"] = [torch.cuda.get_device_name(i) for i in range(info["cudaDeviceCount"])]
            info["defaultNprocPerNode"] = max(1, info["cudaDeviceCount"])
            info["defaultDeviceKind"] = "cuda"
            info["supportedTrainerModes"] = ["baseline", "advanced"]
            info["launchSupport"] = "cuda"
            info["launchMessage"] = f"CUDA available across {info['cudaDeviceCount']} GPU(s)."
    except Exception as exc:
        info["torchError"] = str(exc)

    if system == "Darwin":
        try:
            import mlx  # type: ignore

            info["mlxAvailable"] = True
            info["mlxVersion"] = getattr(mlx, "__version__", None)
            if not info["cudaAvailable"]:
                info["supportedTrainerModes"] = ["baseline"]
                info["defaultDeviceKind"] = "mlx"
                info["launchSupport"] = "mlx"
                info["launchMessage"] = "MLX baseline training is available on this macOS host."
        except Exception as exc:
            info["mlxError"] = str(exc)

    return info


def normalize_env(env: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in env.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            normalized[key] = "1" if value else "0"
        else:
            normalized[key] = str(value)
    return normalized


def resolve_launch_plan(system_info: dict[str, Any], trainer_mode: str, nproc_override: int | None) -> dict[str, Any]:
    if trainer_mode not in {"baseline", "advanced"}:
        raise ValueError(f"trainerMode must be 'baseline' or 'advanced', got {trainer_mode}")

    if trainer_mode == "advanced":
        if not system_info["cudaAvailable"]:
            raise ValueError("Advanced runs require CUDA. No CUDA device was detected on this host.")
        device_count = int(system_info["cudaDeviceCount"])
        nproc = nproc_override or max(1, device_count)
        return {
            "resolvedScriptName": "train_gpt_advanced.py",
            "deviceKind": "cuda",
            "deviceCount": device_count,
            "nprocPerNode": nproc,
            "command": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc}",
                "train_gpt_advanced.py",
            ],
        }

    if system_info["cudaAvailable"]:
        device_count = int(system_info["cudaDeviceCount"])
        nproc = nproc_override or max(1, device_count)
        return {
            "resolvedScriptName": "train_gpt.py",
            "deviceKind": "cuda",
            "deviceCount": device_count,
            "nprocPerNode": nproc,
            "command": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc}",
                "train_gpt.py",
            ],
        }

    if system_info["mlxAvailable"]:
        if nproc_override not in (None, 1):
            raise ValueError("MLX launches do not support NPROC_PER_NODE > 1.")
        return {
            "resolvedScriptName": "train_gpt_mlx.py",
            "deviceKind": "mlx",
            "deviceCount": 1,
            "nprocPerNode": 1,
            "command": [sys.executable, "train_gpt_mlx.py"],
        }

    raise ValueError(
        "No supported accelerator detected. Baseline runs require CUDA on Linux or MLX on macOS."
    )


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ui_{stamp}_{uuid.uuid4().hex[:8]}"


def launch() -> int:
    payload = read_json_stdin()
    trainer_mode = str(payload.get("trainerMode", "baseline"))
    env_overrides = normalize_env(payload.get("env", {}))
    system_info = detect_environment()

    nproc_override_raw = payload.get("nprocPerNode")
    nproc_override = int(nproc_override_raw) if nproc_override_raw not in (None, "") else None
    plan = resolve_launch_plan(system_info, trainer_mode, nproc_override)

    run_id = str(payload.get("runId") or make_run_id())
    experiment_group = payload.get("experimentGroup") or None
    experiment_label = payload.get("experimentLabel") or None
    experiment_comment = payload.get("experimentComment") or None
    launch_id = str(uuid.uuid4())

    env = os.environ.copy()
    env.update(env_overrides)
    env["RUN_ID"] = run_id
    if experiment_group:
        env["EXPERIMENT_GROUP"] = str(experiment_group)
    if experiment_label:
        env["EXPERIMENT_LABEL"] = str(experiment_label)
    if experiment_comment:
        env["EXPERIMENT_COMMENT"] = str(experiment_comment)
    env["LAUNCH_SOURCE"] = "dashboard_ui"
    env["LAUNCH_PLATFORM"] = str(system_info["platform"])
    env["LAUNCH_DEVICE_KIND"] = str(plan["deviceKind"])
    env["LAUNCH_DEVICE_COUNT"] = str(plan["deviceCount"])
    env["LAUNCH_RESOLVED_SCRIPT"] = str(plan["resolvedScriptName"])
    env["LAUNCH_TRAINER_MODE"] = trainer_mode
    env["LAUNCH_ID"] = launch_id

    conn = open_launch_db()
    conn.execute(
        """
        INSERT INTO launch_requests(
            launch_id, run_id, requested_at_utc, updated_at_utc, status, trainer_mode,
            resolved_script_name, platform, device_kind, device_count,
            experiment_group, experiment_label, message, config_json, command_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            launch_id,
            run_id,
            utc_now_iso(),
            utc_now_iso(),
            "launching",
            trainer_mode,
            plan["resolvedScriptName"],
            system_info["platform"],
            plan["deviceKind"],
            plan["deviceCount"],
            experiment_group,
            experiment_label,
            "Preparing background process",
            json.dumps(payload, sort_keys=True),
            json.dumps(plan["command"]),
        ),
    )
    conn.commit()

    logs_dir = ROOT_DIR / "logs" / "launchers"
    logs_dir.mkdir(parents=True, exist_ok=True)
    launcher_log = logs_dir / f"{run_id}.log"
    try:
        with launcher_log.open("ab") as output:
            proc = subprocess.Popen(
                plan["command"],
                cwd=ROOT_DIR,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        conn.execute(
            """
            UPDATE launch_requests
            SET status = ?, updated_at_utc = ?, pid = ?, message = ?
            WHERE launch_id = ?
            """,
            ("spawned", utc_now_iso(), int(proc.pid), f"Launched {plan['resolvedScriptName']}", launch_id),
        )
        conn.commit()
    except Exception as exc:
        conn.execute(
            """
            UPDATE launch_requests
            SET status = ?, updated_at_utc = ?, message = ?
            WHERE launch_id = ?
            """,
            ("failed", utc_now_iso(), str(exc), launch_id),
        )
        conn.commit()
        raise
    finally:
        conn.close()

    result = {
        "launchId": launch_id,
        "runId": run_id,
        "status": "spawned",
        "trainerMode": trainer_mode,
        "resolvedScriptName": plan["resolvedScriptName"],
        "deviceKind": plan["deviceKind"],
        "deviceCount": plan["deviceCount"],
        "nprocPerNode": plan["nprocPerNode"],
        "launcherLogPath": str(launcher_log),
        "message": f"Launched {plan['resolvedScriptName']} in the background.",
    }
    print(json.dumps(result))
    return 0


def system_info() -> int:
    print(json.dumps(detect_environment()))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect the local training environment and launch runs.")
    parser.add_argument("command", choices=("system-info", "launch"))
    args = parser.parse_args()

    try:
        if args.command == "system-info":
            return system_info()
        return launch()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

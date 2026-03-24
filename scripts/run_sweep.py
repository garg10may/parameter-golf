#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_env(env: dict[str, object]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in env.items():
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = "1" if value else "0"
        else:
            normalized[key] = str(value)
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a sequence of experiments from a JSON manifest.")
    parser.add_argument("manifest", help="Path to the sweep manifest JSON file.")
    parser.add_argument("--only", nargs="*", help="Only run experiments with these labels.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep going if one experiment fails.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    root_dir = manifest_path.parent.parent if manifest_path.parent.name == "sweeps" else Path.cwd()
    manifest = load_manifest(manifest_path)

    command = manifest["command"]
    defaults = normalize_env(manifest.get("defaults", {}))
    group = manifest.get("group") or manifest_path.stem
    experiments = manifest["experiments"]
    selected = set(args.only or [])

    failures: list[tuple[str, int]] = []
    print(f"sweep_group={group}")
    print(f"command={command}")
    print(f"experiments={len(experiments)}")

    for index, experiment in enumerate(experiments, start=1):
        label = experiment["label"]
        if selected and label not in selected:
            continue
        exp_env = normalize_env(experiment.get("env", {}))
        run_id = experiment.get("run_id") or f"{group}_{label}_{utc_stamp()}"
        comment = experiment.get("comment", "")
        env = os.environ.copy()
        env.update(defaults)
        env.update(exp_env)
        env["RUN_ID"] = run_id
        env["EXPERIMENT_GROUP"] = group
        env["EXPERIMENT_LABEL"] = label
        env["EXPERIMENT_COMMENT"] = comment

        print("")
        print(f"[{index}/{len(experiments)}] label={label} run_id={run_id}")
        if comment:
            print(f"  comment={comment}")
        if exp_env:
            for key in sorted(exp_env):
                print(f"  {key}={exp_env[key]}")

        result = subprocess.run([command], cwd=root_dir, env=env, check=False)
        if result.returncode != 0:
            failures.append((label, result.returncode))
            print(f"experiment_failed label={label} exit_code={result.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return result.returncode

    if failures:
        print("")
        print("failures:")
        for label, code in failures:
            print(f"  {label}: exit_code={code}")
        return 1

    print("")
    print("sweep_complete")
    print(f"group={group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

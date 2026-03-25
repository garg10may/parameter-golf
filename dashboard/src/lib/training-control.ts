import "server-only";

import path from "node:path";
import { spawnSync } from "node:child_process";
import type { TrainingLaunchResult, TrainingSystemInfo } from "@/lib/types";

function getRepoRoot() {
  return path.resolve(/* turbopackIgnore: true */ process.cwd(), "..");
}

function getLauncherScriptPath() {
  return path.join(getRepoRoot(), "scripts", "training_entrypoint.py");
}

function runLauncher(command: "system-info" | "launch", payload?: unknown) {
  const pythonExecutable = process.env.TRAINING_PYTHON_EXECUTABLE ?? "python3";
  const result = spawnSync(pythonExecutable, [getLauncherScriptPath(), command], {
    cwd: getRepoRoot(),
    input: payload == null ? undefined : JSON.stringify(payload),
    encoding: "utf8",
    env: process.env,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    const stderr = result.stderr?.trim();
    const stdout = result.stdout?.trim();
    throw new Error(stderr || stdout || `Launcher exited with status ${result.status}`);
  }
  const raw = result.stdout?.trim();
  if (!raw) {
    throw new Error("Launcher returned an empty response.");
  }
  return JSON.parse(raw) as unknown;
}

export function getTrainingSystemInfo() {
  return runLauncher("system-info") as TrainingSystemInfo;
}

export function launchTrainingRun(payload: unknown) {
  return runLauncher("launch", payload) as TrainingLaunchResult;
}

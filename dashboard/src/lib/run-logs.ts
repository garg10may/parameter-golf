import "server-only";

import fs from "node:fs";
import { open } from "node:fs/promises";
import path from "node:path";
import { getRunSummary } from "@/lib/queries";

export type RunLogSource = "console" | "run";

export type RunLogSnapshot = {
  runId: string;
  status: string;
  requestedSource: RunLogSource;
  source: RunLogSource | null;
  path: string | null;
  availableSources: Array<{
    key: RunLogSource;
    path: string;
  }>;
  content: string;
  truncated: boolean;
  updatedAtMs: number | null;
};

const DEFAULT_TAIL_BYTES = 128 * 1024;

function getRepoRoot() {
  return path.resolve(/* turbopackIgnore: true */ process.cwd(), "..");
}

function isWithinRepo(candidatePath: string) {
  const repoRoot = getRepoRoot();
  return candidatePath === repoRoot || candidatePath.startsWith(`${repoRoot}${path.sep}`);
}

function resolveRepoPath(filePath: string | null) {
  if (!filePath) {
    return null;
  }
  const resolved = path.resolve(getRepoRoot(), filePath);
  return isWithinRepo(resolved) ? resolved : null;
}

function resolveConsoleLogPath(runId: string) {
  return path.join(getRepoRoot(), "logs", "launchers", `${runId}.log`);
}

function getAvailableSources(runId: string, runLogPath: string | null) {
  const sources: Array<{ key: RunLogSource; path: string }> = [];
  const consolePath = resolveConsoleLogPath(runId);
  if (fs.existsSync(consolePath)) {
    sources.push({ key: "console", path: consolePath });
  }
  const resolvedRunLogPath = resolveRepoPath(runLogPath);
  if (resolvedRunLogPath && fs.existsSync(resolvedRunLogPath)) {
    sources.push({ key: "run", path: resolvedRunLogPath });
  }
  return sources;
}

async function readTail(filePath: string, maxBytes: number) {
  const handle = await open(filePath, "r");
  try {
    const stats = await handle.stat();
    const size = Number(stats.size);
    const length = Math.min(size, maxBytes);
    if (length <= 0) {
      return {
        content: "",
        truncated: false,
        updatedAtMs: stats.mtimeMs,
      };
    }

    const start = size - length;
    const buffer = Buffer.alloc(length);
    await handle.read(buffer, 0, length, start);

    let content = buffer.toString("utf8");
    const truncated = start > 0;
    if (truncated) {
      const firstNewline = content.indexOf("\n");
      if (firstNewline >= 0 && firstNewline < content.length - 1) {
        content = content.slice(firstNewline + 1);
      }
    }

    return {
      content,
      truncated,
      updatedAtMs: stats.mtimeMs,
    };
  } finally {
    await handle.close();
  }
}

export async function getRunLogSnapshot(runId: string, requestedSource: RunLogSource = "console"): Promise<RunLogSnapshot | null> {
  const summary = getRunSummary(runId);
  if (!summary) {
    return null;
  }

  const availableSources = getAvailableSources(runId, summary.logPath);
  const preferredSource =
    availableSources.find((source) => source.key === requestedSource) ??
    availableSources[0] ??
    null;

  if (!preferredSource) {
    return {
      runId,
      status: summary.status,
      requestedSource,
      source: null,
      path: null,
      availableSources,
      content: "",
      truncated: false,
      updatedAtMs: null,
    };
  }

  const snapshot = await readTail(preferredSource.path, DEFAULT_TAIL_BYTES);
  return {
    runId,
    status: summary.status,
    requestedSource,
    source: preferredSource.key,
    path: preferredSource.path,
    availableSources,
    content: snapshot.content,
    truncated: snapshot.truncated,
    updatedAtMs: snapshot.updatedAtMs,
  };
}

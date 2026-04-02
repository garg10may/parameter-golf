"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cn, formatRelativeDate, toTitleCase } from "@/lib/format";

type RunLogSource = "console" | "run";

type RunLogSnapshot = {
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

const LIVE_STATUSES = new Set(["launching", "spawned", "running"]);

function sourceLabel(source: RunLogSource) {
  return source === "console" ? "Console log" : "Run log";
}

export function RunLogPanel({
  runId,
  initialStatus,
}: {
  runId: string;
  initialStatus: string;
}) {
  const [selectedSource, setSelectedSource] = useState<RunLogSource>("console");
  const [snapshot, setSnapshot] = useState<RunLogSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const logContainerRef = useRef<HTMLPreElement | null>(null);

  const refreshSnapshot = useCallback(
    async (requestedSource: RunLogSource, background = false) => {
      if (!background) {
        setIsLoading(true);
      }
      try {
        const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/log?source=${requestedSource}`, {
          cache: "no-store",
        });
        const data = (await response.json()) as RunLogSnapshot | { error?: string };
        if (!response.ok) {
          throw new Error("error" in data && data.error ? data.error : "Failed to load run log.");
        }
        setSnapshot(data as RunLogSnapshot);
        setError(null);
      } catch (nextError) {
        setError(nextError instanceof Error ? nextError.message : "Failed to load run log.");
      } finally {
        if (!background) {
          setIsLoading(false);
        }
      }
    },
    [runId],
  );

  useEffect(() => {
    void refreshSnapshot(selectedSource, false);
  }, [refreshSnapshot, selectedSource]);

  const liveStatus = snapshot?.status ?? initialStatus;
  const shouldPoll = autoRefresh && LIVE_STATUSES.has(liveStatus);

  useEffect(() => {
    if (!shouldPoll) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void refreshSnapshot(selectedSource, true);
    }, 2000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [refreshSnapshot, selectedSource, shouldPoll]);

  useEffect(() => {
    if (!logContainerRef.current) {
      return;
    }
    logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
  }, [snapshot?.content]);

  const activeSource = snapshot?.source ?? selectedSource;
  const availableSources = snapshot?.availableSources ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-2">
          {(["console", "run"] as const).map((source) => {
            const available = availableSources.some((item) => item.key === source);
            return (
              <button
                key={source}
                type="button"
                onClick={() => setSelectedSource(source)}
                disabled={isLoading || (!available && snapshot != null)}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-sm transition",
                  activeSource === source
                    ? "border-stone-900 bg-stone-900 text-white"
                    : "border-stone-300 bg-white text-stone-700 hover:border-stone-500",
                  !available && snapshot != null && "cursor-not-allowed border-stone-200 bg-stone-100 text-stone-400 hover:border-stone-200",
                )}
              >
                {sourceLabel(source)}
              </button>
            );
          })}
        </div>
        <div className="flex flex-wrap items-center gap-3 text-sm text-stone-600">
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => setAutoRefresh(event.target.checked)}
              className="h-4 w-4 rounded border-stone-300 text-stone-900"
            />
            Live refresh
          </label>
          <button
            type="button"
            onClick={() => void refreshSnapshot(selectedSource, false)}
            className="rounded-full border border-stone-300 bg-white px-3 py-1.5 text-sm text-stone-700 transition hover:border-stone-500"
          >
            Refresh
          </button>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 rounded-[24px] border border-stone-200 bg-stone-50/90 px-4 py-3 text-sm text-stone-600">
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-medium text-stone-900">{sourceLabel(activeSource)}</span>
          <span>Status: {toTitleCase(liveStatus)}</span>
          {snapshot?.updatedAtMs ? <span>Updated {formatRelativeDate(new Date(snapshot.updatedAtMs).toISOString())}</span> : null}
        </div>
        {snapshot?.path ? <div className="break-all font-mono text-[11px] text-stone-500">{snapshot.path}</div> : null}
      </div>

      {error ? (
        <div className="rounded-[24px] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">
          {error}
        </div>
      ) : null}

      {snapshot && !snapshot.source ? (
        <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
          No log file is available for this run yet.
        </div>
      ) : (
        <div className="overflow-hidden rounded-[24px] border border-stone-200 bg-stone-950">
          <pre
            ref={logContainerRef}
            className="max-h-[560px] overflow-auto px-4 py-4 font-mono text-[12px] leading-5 text-stone-100"
          >
            {isLoading && !snapshot ? "Loading log output..." : snapshot?.content || "Log file exists but is still empty."}
          </pre>
        </div>
      )}

      {snapshot?.truncated ? (
        <div className="text-xs text-stone-500">Showing the latest tail of the file, not the full log.</div>
      ) : null}
    </div>
  );
}

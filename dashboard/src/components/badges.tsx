import { cn, formatMetricLabel, formatPhaseLabel } from "@/lib/format";
import type { RunSummary } from "@/lib/types";

export function ProxyBadge({ run }: { run: RunSummary }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full px-2.5 py-1 text-[11px] font-medium",
        run.trusted
          ? "bg-emerald-100 text-emerald-900"
          : "bg-amber-100 text-amber-900",
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", run.trusted ? "bg-emerald-500" : "bg-amber-500")} />
      {run.trusted ? "Trusted" : "Proxy only"}
    </span>
  );
}

export function MetricBadge({ run }: { run: RunSummary }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-stone-200 bg-stone-100 px-2.5 py-1 text-[11px] font-medium text-stone-700">
      {formatMetricLabel(run.bestAvailableMetricName)}
      <span className="text-stone-400">·</span>
      {formatPhaseLabel(run.bestAvailableMetricPhase)}
    </span>
  );
}

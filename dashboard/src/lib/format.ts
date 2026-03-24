import { type BestMetricName, type MetricPhase, type RunSummary } from "@/lib/types";

export function cn(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export function formatNumber(value: number | null | undefined, digits = 4) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

export function formatCompactNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatInteger(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return Math.round(value).toLocaleString();
}

export function formatPercent(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

export function formatBytes(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return "—";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let idx = 0;
  while (size >= 1000 && idx < units.length - 1) {
    size /= 1000;
    idx += 1;
  }
  return `${size.toFixed(idx === 0 ? 0 : 2)} ${units[idx]}`;
}

export function formatDuration(ms: number | null | undefined) {
  if (ms == null || Number.isNaN(ms)) {
    return "—";
  }
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

export function formatRelativeDate(value: string | null | undefined) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function formatMetricLabel(name: BestMetricName | null | undefined) {
  if (name === "roundtrip_val_bpb") return "Roundtrip val_bpb";
  if (name === "val_bpb") return "Validation val_bpb";
  if (name === "train_loss") return "Train loss";
  return "Metric";
}

export function formatPhaseLabel(phase: MetricPhase | null | undefined) {
  if (phase === "roundtrip") return "Trusted";
  if (phase === "val") return "Validated";
  if (phase === "train") return "Proxy";
  return "Unknown";
}

export function metricDirectionHint(metricName: BestMetricName | null | undefined) {
  if (!metricName) {
    return "Lower is better";
  }
  return metricName.endsWith("bpb") || metricName === "train_loss" ? "Lower is better" : "";
}

export function deriveMetricAccent(run: RunSummary) {
  if (run.bestAvailableMetricPhase === "roundtrip") {
    return "trusted";
  }
  if (run.bestAvailableMetricPhase === "val") {
    return "validated";
  }
  return "proxy";
}

export function formatModelFootprint(run: Pick<RunSummary, "modelParamCount" | "modelDim" | "numLayers" | "numHeads" | "numKvHeads" | "mlpMult">) {
  if (run.modelParamCount != null && !Number.isNaN(run.modelParamCount)) {
    return `${formatCompactNumber(run.modelParamCount)} params`;
  }

  const parts = [
    run.modelDim != null ? `${run.modelDim}d` : null,
    run.numLayers != null ? `${run.numLayers}L` : null,
    run.numHeads != null ? `${run.numHeads}H` : null,
    run.numKvHeads != null ? `${run.numKvHeads}KV` : null,
    run.mlpMult != null ? `MLP×${run.mlpMult}` : null,
  ].filter(Boolean);

  if (parts.length === 0) {
    return "—";
  }

  return parts.join(" · ");
}

export function toTitleCase(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

import "server-only";

import { getDatabasePath, getDb } from "@/lib/db";
import type {
  BestMetricName,
  GroupDeltaItem,
  GroupDetail,
  GroupSummary,
  MetricPhase,
  MetricSeries,
  OverviewData,
  ParameterImpactPoint,
  ParameterOption,
  RunArtifact,
  RunDetail,
  RunEvent,
  RunFilters,
  RunParameter,
  RunSummary,
} from "@/lib/types";

const METRIC_PRIORITY: Array<{ name: BestMetricName; phase: MetricPhase }> = [
  { name: "roundtrip_val_bpb", phase: "roundtrip" },
  { name: "val_bpb", phase: "val" },
  { name: "train_loss", phase: "train" },
];

const EXCLUDED_PARAM_NAMES = new Set([
  "actual_train_shards",
  "data_path",
  "dataset_name",
  "expected_train_shards",
  "experiment_comment",
  "experiment_group",
  "experiment_label",
  "model_param_count",
  "out_dir",
  "run_id",
  "script_name",
  "tokenizer_path",
  "train_files",
  "val_files",
]);

type RawRunRow = {
  run_id: string;
  script_name: string;
  backend: string;
  status: string;
  started_at_utc: string;
  finished_at_utc: string | null;
  log_path: string | null;
  notes: string | null;
  experiment_group: string | null;
  experiment_label: string | null;
  experiment_comment: string | null;
  model_param_count: number | null;
  model_dim: number | null;
  num_layers: number | null;
  num_heads: number | null;
  num_kv_heads: number | null;
  mlp_mult: number | null;
  rope_base: number | null;
  tied_embed_lr: number | null;
  matrix_lr: number | null;
  seed: number | null;
  final_train_loss: number | null;
  final_train_time_ms: number | null;
  final_tok_s: number | null;
  avg_tok_s: number | null;
  final_val_loss: number | null;
  final_val_bpb: number | null;
  final_roundtrip_val_loss: number | null;
  final_roundtrip_val_bpb: number | null;
  max_step: number | null;
  artifact_bytes: number | null;
  artifact_name: string | null;
};

function deriveRunDurationMs(row: RawRunRow) {
  if (row.finished_at_utc) {
    const started = Date.parse(row.started_at_utc);
    const finished = Date.parse(row.finished_at_utc);
    if (!Number.isNaN(started) && !Number.isNaN(finished)) {
      return Math.max(0, finished - started);
    }
  }
  return row.final_train_time_ms ?? null;
}

function deriveBestMetric(row: RawRunRow) {
  if (row.final_roundtrip_val_bpb != null) {
    return {
      bestAvailableMetricName: "roundtrip_val_bpb" as const,
      bestAvailableMetricValue: row.final_roundtrip_val_bpb,
      bestAvailableMetricPhase: "roundtrip" as const,
      proxyOnly: false,
      trusted: true,
    };
  }
  if (row.final_val_bpb != null) {
    return {
      bestAvailableMetricName: "val_bpb" as const,
      bestAvailableMetricValue: row.final_val_bpb,
      bestAvailableMetricPhase: "val" as const,
      proxyOnly: false,
      trusted: true,
    };
  }
  if (row.final_train_loss != null) {
    return {
      bestAvailableMetricName: "train_loss" as const,
      bestAvailableMetricValue: row.final_train_loss,
      bestAvailableMetricPhase: "train" as const,
      proxyOnly: true,
      trusted: false,
    };
  }
  return {
    bestAvailableMetricName: null,
    bestAvailableMetricValue: null,
    bestAvailableMetricPhase: null,
    proxyOnly: true,
    trusted: false,
  };
}

function mapRunRow(row: RawRunRow): RunSummary {
  return {
    runId: row.run_id,
    scriptName: row.script_name,
    backend: row.backend,
    status: row.status,
    startedAtUtc: row.started_at_utc,
    finishedAtUtc: row.finished_at_utc,
    logPath: row.log_path,
    notes: row.notes,
    experimentGroup: row.experiment_group,
    experimentLabel: row.experiment_label,
    experimentComment: row.experiment_comment,
    modelParamCount: row.model_param_count,
    modelDim: row.model_dim,
    numLayers: row.num_layers,
    numHeads: row.num_heads,
    numKvHeads: row.num_kv_heads,
    mlpMult: row.mlp_mult,
    ropeBase: row.rope_base,
    tiedEmbedLr: row.tied_embed_lr,
    matrixLr: row.matrix_lr,
    seed: row.seed,
    finalTrainLoss: row.final_train_loss,
    finalTrainTimeMs: row.final_train_time_ms,
    finalTokS: row.final_tok_s,
    avgTokS: row.avg_tok_s,
    finalValLoss: row.final_val_loss,
    finalValBpb: row.final_val_bpb,
    finalRoundtripValLoss: row.final_roundtrip_val_loss,
    finalRoundtripValBpb: row.final_roundtrip_val_bpb,
    maxStep: row.max_step,
    artifactBytes: row.artifact_bytes,
    artifactName: row.artifact_name,
    runDurationMs: deriveRunDurationMs(row),
    ...deriveBestMetric(row),
  };
}

function metricPriorityIndex(run: RunSummary) {
  if (run.bestAvailableMetricName == null) {
    return 999;
  }
  return METRIC_PRIORITY.findIndex((item) => item.name === run.bestAvailableMetricName);
}

function compareRunsByBestMetric(a: RunSummary, b: RunSummary) {
  const aPriority = metricPriorityIndex(a);
  const bPriority = metricPriorityIndex(b);
  if (aPriority !== bPriority) {
    return aPriority - bPriority;
  }
  if (a.bestAvailableMetricValue != null && b.bestAvailableMetricValue != null) {
    return a.bestAvailableMetricValue - b.bestAvailableMetricValue;
  }
  if (a.bestAvailableMetricValue != null) return -1;
  if (b.bestAvailableMetricValue != null) return 1;
  return b.startedAtUtc.localeCompare(a.startedAtUtc);
}

function pickBestRun(runs: RunSummary[]) {
  if (runs.length === 0) {
    return null;
  }
  return [...runs].sort(compareRunsByBestMetric)[0] ?? null;
}

function pickPreferredMetricName(runs: RunSummary[]): BestMetricName | null {
  for (const metric of METRIC_PRIORITY) {
    if (
      runs.some((run) => {
        if (metric.name === "roundtrip_val_bpb") return run.finalRoundtripValBpb != null;
        if (metric.name === "val_bpb") return run.finalValBpb != null;
        return run.finalTrainLoss != null;
      })
    ) {
      return metric.name;
    }
  }
  return null;
}

function getMetricValueForName(run: RunSummary, metricName: BestMetricName) {
  if (metricName === "roundtrip_val_bpb") return run.finalRoundtripValBpb;
  if (metricName === "val_bpb") return run.finalValBpb;
  return run.finalTrainLoss;
}

function baseRunQuery() {
  return `
    WITH ranked_metrics AS (
      SELECT
        run_id,
        phase,
        name,
        value,
        step,
        id,
        ROW_NUMBER() OVER (
          PARTITION BY run_id, phase, name
          ORDER BY COALESCE(step, -1) DESC, id DESC
        ) AS rn
      FROM metrics
    ),
    latest_metrics AS (
      SELECT
        run_id,
        MAX(CASE WHEN phase = 'train' AND name = 'train_loss' AND rn = 1 THEN value END) AS final_train_loss,
        MAX(CASE WHEN phase = 'train' AND name = 'train_time_ms' AND rn = 1 THEN value END) AS final_train_time_ms,
        MAX(CASE WHEN phase = 'train' AND name = 'tok_s' AND rn = 1 THEN value END) AS final_tok_s,
        AVG(CASE WHEN phase = 'train' AND name = 'tok_s' THEN value END) AS avg_tok_s,
        MAX(CASE WHEN phase = 'val' AND name = 'val_loss' AND rn = 1 THEN value END) AS final_val_loss,
        MAX(CASE WHEN phase = 'val' AND name = 'val_bpb' AND rn = 1 THEN value END) AS final_val_bpb,
        MAX(CASE WHEN phase = 'roundtrip' AND name = 'val_loss' AND rn = 1 THEN value END) AS final_roundtrip_val_loss,
        MAX(CASE WHEN phase = 'roundtrip' AND name = 'val_bpb' AND rn = 1 THEN value END) AS final_roundtrip_val_bpb,
        MAX(CASE WHEN step IS NOT NULL THEN step END) AS max_step
      FROM ranked_metrics
      GROUP BY run_id
    ),
    pivot_params AS (
      SELECT
        run_id,
        MAX(CASE WHEN name = 'experiment_group' THEN value_text END) AS experiment_group,
        MAX(CASE WHEN name = 'experiment_label' THEN value_text END) AS experiment_label,
        MAX(CASE WHEN name = 'experiment_comment' THEN value_text END) AS experiment_comment,
        MAX(CASE WHEN name = 'model_param_count' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS model_param_count,
        MAX(CASE WHEN name = 'model_dim' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS model_dim,
        MAX(CASE WHEN name = 'num_layers' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS num_layers,
        MAX(CASE WHEN name = 'num_heads' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS num_heads,
        MAX(CASE WHEN name = 'num_kv_heads' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS num_kv_heads,
        MAX(CASE WHEN name = 'mlp_mult' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS mlp_mult,
        MAX(CASE WHEN name = 'rope_base' THEN COALESCE(value_real, CAST(value_int AS REAL)) END) AS rope_base,
        MAX(CASE WHEN name = 'tied_embed_lr' THEN COALESCE(value_real, CAST(value_int AS REAL)) END) AS tied_embed_lr,
        MAX(CASE WHEN name = 'matrix_lr' THEN COALESCE(value_real, CAST(value_int AS REAL)) END) AS matrix_lr,
        MAX(CASE WHEN name = 'seed' THEN COALESCE(value_int, CAST(value_real AS INTEGER)) END) AS seed
      FROM params
      GROUP BY run_id
    ),
    artifact_summary AS (
      SELECT
        run_id,
        MAX(bytes) AS artifact_bytes,
        MAX(name) AS artifact_name
      FROM artifacts
      GROUP BY run_id
    )
    SELECT
      r.run_id,
      r.script_name,
      r.backend,
      r.status,
      r.started_at_utc,
      r.finished_at_utc,
      r.log_path,
      r.notes,
      p.experiment_group,
      p.experiment_label,
      p.experiment_comment,
      p.model_param_count,
      p.model_dim,
      p.num_layers,
      p.num_heads,
      p.num_kv_heads,
      p.mlp_mult,
      p.rope_base,
      p.tied_embed_lr,
      p.matrix_lr,
      p.seed,
      m.final_train_loss,
      m.final_train_time_ms,
      m.final_tok_s,
      m.avg_tok_s,
      m.final_val_loss,
      m.final_val_bpb,
      m.final_roundtrip_val_loss,
      m.final_roundtrip_val_bpb,
      m.max_step,
      a.artifact_bytes,
      a.artifact_name
    FROM runs r
    LEFT JOIN latest_metrics m ON m.run_id = r.run_id
    LEFT JOIN pivot_params p ON p.run_id = r.run_id
    LEFT JOIN artifact_summary a ON a.run_id = r.run_id
  `;
}

export function getAllRunSummaries(): RunSummary[] {
  const db = getDb();
  if (!db) {
    return [];
  }
  const rows = db.prepare(`${baseRunQuery()} ORDER BY r.started_at_utc DESC`).all() as RawRunRow[];
  return rows.map(mapRunRow);
}

function filterRuns(runs: RunSummary[], filters: RunFilters) {
  return runs.filter((run) => {
    if (filters.backend && filters.backend !== "all" && run.backend !== filters.backend) return false;
    if (filters.status && filters.status !== "all" && run.status !== filters.status) return false;
    if (filters.group && filters.group !== "all" && run.experimentGroup !== filters.group) return false;
    if (filters.phase && filters.phase !== "all" && run.bestAvailableMetricPhase !== filters.phase) return false;
    if (filters.from && run.startedAtUtc.slice(0, 10) < filters.from) return false;
    if (filters.to && run.startedAtUtc.slice(0, 10) > filters.to) return false;
    if (filters.query) {
      const haystack = [
        run.runId,
        run.experimentGroup,
        run.experimentLabel,
        run.experimentComment,
        run.notes,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(filters.query.toLowerCase())) return false;
    }
    return true;
  });
}

function sortRuns(runs: RunSummary[], sort = "started", dir: "asc" | "desc" = "desc") {
  const factor = dir === "asc" ? 1 : -1;
  return [...runs].sort((a, b) => {
    let delta = 0;
    switch (sort) {
      case "best_metric":
        delta = compareRunsByBestMetric(a, b);
        break;
      case "train_loss":
        delta = (a.finalTrainLoss ?? Number.POSITIVE_INFINITY) - (b.finalTrainLoss ?? Number.POSITIVE_INFINITY);
        break;
      case "duration":
        delta = (a.runDurationMs ?? Number.POSITIVE_INFINITY) - (b.runDurationMs ?? Number.POSITIVE_INFINITY);
        break;
      case "params":
        delta = (a.modelParamCount ?? 0) - (b.modelParamCount ?? 0);
        break;
      case "started":
      default:
        delta = a.startedAtUtc.localeCompare(b.startedAtUtc);
        break;
    }
    return delta * factor;
  });
}

export function getRunSummaries(filters: RunFilters = {}) {
  const runs = getAllRunSummaries();
  const filtered = filterRuns(runs, filters);
  return sortRuns(filtered, filters.sort, filters.dir);
}

export function getRunSummary(runId: string) {
  const db = getDb();
  if (!db) return null;
  const row = db.prepare(`${baseRunQuery()} WHERE r.run_id = ?`).get(runId) as RawRunRow | undefined;
  return row ? mapRunRow(row) : null;
}

function readParamValue(row: { value_type: string; value_text: string | null; value_real: number | null; value_int: number | null }) {
  if (row.value_type === "bool") return Boolean(row.value_int);
  if (row.value_type === "int") return row.value_int;
  if (row.value_type === "float") return row.value_real;
  return row.value_text;
}

export function getRunParameters(runId: string): RunParameter[] {
  const db = getDb();
  if (!db) return [];
  const rows = db
    .prepare(
      `
      SELECT name, value_type, value_text, value_real, value_int
      FROM params
      WHERE run_id = ?
      ORDER BY name ASC
      `,
    )
    .all(runId) as Array<{ name: string; value_type: string; value_text: string | null; value_real: number | null; value_int: number | null }>;
  return rows.map((row) => ({
    name: row.name,
    valueType: row.value_type,
    value: readParamValue(row),
  }));
}

export function getRunMetricSeries(runId: string): MetricSeries[] {
  const db = getDb();
  if (!db) return [];
  const rows = db
    .prepare(
      `
      SELECT phase, name, COALESCE(step, 0) AS step, value
      FROM metrics
      WHERE run_id = ?
      ORDER BY phase ASC, name ASC, COALESCE(step, 0) ASC, id ASC
      `,
    )
    .all(runId) as Array<{ phase: string; name: string; step: number; value: number }>;
  const grouped = new Map<string, MetricSeries>();
  for (const row of rows) {
    const key = `${row.phase}.${row.name}`;
    const existing = grouped.get(key) ?? {
      key,
      phase: row.phase,
      name: row.name,
      points: [],
    };
    existing.points.push({ step: row.step, value: row.value });
    grouped.set(key, existing);
  }
  return [...grouped.values()];
}

export function getRunEvents(runId: string): RunEvent[] {
  const db = getDb();
  if (!db) return [];
  return db
    .prepare(
      `
      SELECT id, step, name, message, recorded_at_utc
      FROM events
      WHERE run_id = ?
      ORDER BY recorded_at_utc DESC, id DESC
      `,
    )
    .all(runId) as RunEvent[];
}

export function getRunArtifacts(runId: string): RunArtifact[] {
  const db = getDb();
  if (!db) return [];
  return db
    .prepare(
      `
      SELECT id, name, bytes, metadata_json, recorded_at_utc
      FROM artifacts
      WHERE run_id = ?
      ORDER BY recorded_at_utc DESC, id DESC
      `,
    )
    .all(runId) as RunArtifact[];
}

export function getRunDetail(runId: string): RunDetail | null {
  const summary = getRunSummary(runId);
  if (!summary) {
    return null;
  }
  return {
    summary,
    params: getRunParameters(runId),
    metrics: getRunMetricSeries(runId),
    events: getRunEvents(runId),
    artifacts: getRunArtifacts(runId),
  };
}

export function getGroupSummaries(): GroupSummary[] {
  const runs = getAllRunSummaries().filter((run) => run.experimentGroup);
  const groups = new Map<string, RunSummary[]>();
  for (const run of runs) {
    const group = run.experimentGroup!;
    const list = groups.get(group) ?? [];
    list.push(run);
    groups.set(group, list);
  }
  return [...groups.entries()]
    .map(([name, groupRuns]) => {
      const latestRun = [...groupRuns].sort((a, b) => b.startedAtUtc.localeCompare(a.startedAtUtc))[0]!;
      return {
        name,
        runCount: groupRuns.length,
        latestRunId: latestRun.runId,
        latestStartedAtUtc: latestRun.startedAtUtc,
        baselinePresent: groupRuns.some((run) => run.experimentLabel === "baseline"),
        bestRun: pickBestRun(groupRuns)!,
      };
    })
    .sort((a, b) => b.latestStartedAtUtc.localeCompare(a.latestStartedAtUtc));
}

function getChangedParamsForRuns(runIds: string[]) {
  const db = getDb();
  if (!db || runIds.length === 0) return [];
  const placeholders = runIds.map(() => "?").join(", ");
  const rows = db
    .prepare(
      `
      SELECT
        name,
        COUNT(DISTINCT COALESCE(value_text, CAST(value_int AS TEXT), CAST(value_real AS TEXT))) AS distinct_values
      FROM params
      WHERE run_id IN (${placeholders})
        AND name NOT IN (${[...EXCLUDED_PARAM_NAMES].map(() => "?").join(", ")})
      GROUP BY name
      HAVING distinct_values > 1
      ORDER BY name ASC
      `,
    )
    .all(...runIds, ...EXCLUDED_PARAM_NAMES) as Array<{ name: string }>;
  return rows.map((row) => row.name);
}

export function getGroupDetail(groupName: string): GroupDetail | null {
  const runs = getAllRunSummaries().filter((run) => run.experimentGroup === groupName);
  if (runs.length === 0) {
    return null;
  }
  const summary = getGroupSummaries().find((group) => group.name === groupName)!;
  const baseline = runs.find((run) => run.experimentLabel === "baseline") ?? null;
  const metricName = pickPreferredMetricName(runs);
  const deltas: GroupDeltaItem[] = [];
  if (baseline && metricName) {
    const baselineValue = getMetricValueForName(baseline, metricName);
    if (baselineValue != null) {
      for (const run of runs) {
        const metricValue = getMetricValueForName(run, metricName);
        if (metricValue == null) continue;
        deltas.push({
          runId: run.runId,
          label: run.experimentLabel ?? run.runId,
          comment: run.experimentComment,
          metricName,
          metricValue,
          baselineValue,
          delta: metricValue - baselineValue,
          trusted: run.trusted,
          proxyOnly: run.proxyOnly,
        });
      }
    }
  }
  const trainCurves = runs
    .map((run) => {
      const series = getRunMetricSeries(run.runId).find((item) => item.phase === "train" && item.name === "train_loss");
      return {
        runId: run.runId,
        label: run.experimentLabel ?? run.runId,
        points: series?.points ?? [],
      };
    })
    .filter((item) => item.points.length > 0);

  return {
    summary,
    runs: [...runs].sort(compareRunsByBestMetric),
    baseline,
    changedParams: getChangedParamsForRuns(runs.map((run) => run.runId)),
    deltas: deltas.sort((a, b) => a.delta - b.delta),
    trainCurves,
  };
}

export function getParameterOptions(): ParameterOption[] {
  const db = getDb();
  if (!db) return [];
  const rows = db
    .prepare(
      `
      SELECT
        name,
        COUNT(*) AS run_count,
        COUNT(DISTINCT COALESCE(value_text, CAST(value_int AS TEXT), CAST(value_real AS TEXT))) AS distinct_value_count,
        SUM(CASE WHEN value_int IS NOT NULL OR value_real IS NOT NULL THEN 1 ELSE 0 END) AS numeric_rows
      FROM params
      WHERE name NOT IN (${[...EXCLUDED_PARAM_NAMES].map(() => "?").join(", ")})
      GROUP BY name
      HAVING distinct_value_count > 1
      ORDER BY distinct_value_count DESC, run_count DESC, name ASC
      `,
    )
    .all(...EXCLUDED_PARAM_NAMES) as Array<{
      name: string;
      run_count: number;
      distinct_value_count: number;
      numeric_rows: number;
    }>;
  return rows.map((row) => ({
    name: row.name,
    distinctValueCount: row.distinct_value_count,
    numeric: row.numeric_rows === row.run_count,
    runCount: row.run_count,
  }));
}

export function getParameterImpact(paramName: string | null): ParameterImpactPoint[] {
  if (!paramName) return [];
  const db = getDb();
  if (!db) return [];
  const runs = getAllRunSummaries();
  const preferredMetricName = pickPreferredMetricName(runs);
  if (!preferredMetricName) return [];
  const rows = db
    .prepare(
      `
      SELECT run_id, value_type, value_text, value_real, value_int
      FROM params
      WHERE name = ?
      `,
    )
    .all(paramName) as Array<{
      run_id: string;
      value_type: string;
      value_text: string | null;
      value_real: number | null;
      value_int: number | null;
    }>;
  const counts = new Map<string, number>();
  const lookup = new Map<string, { value: string | number | boolean | null; numericValue: number | null }>();
  for (const row of rows) {
    const value = readParamValue(row);
    const numericValue =
      typeof value === "number"
        ? value
        : row.value_real != null
          ? row.value_real
          : row.value_int != null
            ? row.value_int
            : null;
    lookup.set(row.run_id, { value, numericValue });
    const countKey = String(value);
    counts.set(countKey, (counts.get(countKey) ?? 0) + 1);
  }
  return runs
    .map((run) => {
      const param = lookup.get(run.runId);
      if (!param) return null;
      const metricValue = getMetricValueForName(run, preferredMetricName);
      if (metricValue == null || run.bestAvailableMetricPhase == null) return null;
      return {
        runId: run.runId,
        label: run.experimentLabel ?? run.runId,
        group: run.experimentGroup,
        value: param.value,
        numericValue: param.numericValue,
        metricName: preferredMetricName,
        metricPhase: run.bestAvailableMetricPhase,
        metricValue,
        trusted: run.trusted,
        proxyOnly: run.proxyOnly,
        sampleCount: counts.get(String(param.value)) ?? 1,
      } satisfies ParameterImpactPoint;
    })
    .filter((item): item is ParameterImpactPoint => Boolean(item));
}

function buildSuspiciousRuns(runs: RunSummary[]) {
  const baselines = new Map<string, RunSummary>();
  for (const run of runs) {
    if (run.experimentGroup && run.experimentLabel === "baseline") {
      baselines.set(run.experimentGroup, run);
    }
  }

  const medianTokS = [...runs]
    .map((run) => run.finalTokS ?? run.avgTokS)
    .filter((value): value is number => value != null)
    .sort((a, b) => a - b);
  const medianModelSize = [...runs]
    .map((run) => run.modelParamCount)
    .filter((value): value is number => value != null)
    .sort((a, b) => a - b);
  const tokSMedian = medianTokS.length > 0 ? medianTokS[Math.floor(medianTokS.length / 2)] : null;
  const modelMedian = medianModelSize.length > 0 ? medianModelSize[Math.floor(medianModelSize.length / 2)] : null;

  const suspicious = [];
  for (const run of runs) {
    if (run.proxyOnly && tokSMedian != null && (run.finalTokS ?? 0) > tokSMedian * 1.8) {
      suspicious.push({
        run,
        reason: "Train-only proxy is unusually fast. Treat ranking as optimization-speed signal, not quality.",
      });
      continue;
    }
    if (run.proxyOnly && modelMedian != null && (run.modelParamCount ?? modelMedian) < modelMedian * 0.4) {
      suspicious.push({
        run,
        reason: "Tiny train-only run needs validation before it can be compared to larger trusted runs.",
      });
      continue;
    }
    if (run.experimentGroup && baselines.has(run.experimentGroup) && run.experimentLabel !== "baseline") {
      const baseline = baselines.get(run.experimentGroup)!;
      if (
        baseline.bestAvailableMetricName === run.bestAvailableMetricName &&
        baseline.bestAvailableMetricValue != null &&
        run.bestAvailableMetricValue != null &&
        run.bestAvailableMetricValue - baseline.bestAvailableMetricValue > 0.12
      ) {
        suspicious.push({
          run,
          reason: "Regressed sharply relative to the group baseline under the same best-available metric.",
        });
      }
    }
  }
  return suspicious.slice(0, 8);
}

function buildSweepDeltas(runs: RunSummary[]) {
  const grouped = new Map<string, RunSummary[]>();
  for (const run of runs) {
    if (!run.experimentGroup) continue;
    const list = grouped.get(run.experimentGroup) ?? [];
    list.push(run);
    grouped.set(run.experimentGroup, list);
  }
  const items: OverviewData["sweepDeltas"] = [];
  for (const [group, groupRuns] of grouped) {
    const baseline = groupRuns.find((run) => run.experimentLabel === "baseline");
    if (!baseline) continue;
    const metricName = pickPreferredMetricName(groupRuns);
    if (!metricName) continue;
    const baselineValue = getMetricValueForName(baseline, metricName);
    if (baselineValue == null) continue;
    for (const run of groupRuns) {
      if (run.runId === baseline.runId) continue;
      const metricValue = getMetricValueForName(run, metricName);
      if (metricValue == null) continue;
      items.push({
        group,
        label: run.experimentLabel ?? run.runId,
        metricName,
        metricValue,
        baselineValue,
        delta: metricValue - baselineValue,
        proxyOnly: run.proxyOnly,
        trusted: run.trusted,
      });
    }
  }
  return items.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 16);
}

export function getOverviewData(selectedParameter?: string | null): OverviewData {
  const runs = getAllRunSummaries();
  const parameterOptions = getParameterOptions();
  const parameterFallback =
    selectedParameter ??
    ["model_dim", "rope_base", "num_kv_heads", "mlp_mult", "tied_embed_lr", "matrix_lr"].find((name) =>
      parameterOptions.some((option) => option.name === name),
    ) ??
    parameterOptions[0]?.name ??
    null;
  const grouped = getGroupSummaries();
  const distinctKnobCount = parameterOptions.length;
  return {
    dbPath: getDatabasePath(),
    runCount: runs.length,
    groupCount: grouped.length,
    distinctKnobCount,
    bestRoundtripRun: pickBestRun(runs.filter((run) => run.finalRoundtripValBpb != null)),
    bestValidationRun: pickBestRun(runs.filter((run) => run.finalRoundtripValBpb == null && run.finalValBpb != null)),
    bestProxyRun: pickBestRun(runs.filter((run) => run.proxyOnly && run.finalTrainLoss != null)),
    bestRecentRuns: [...runs].sort(compareRunsByBestMetric).slice(0, 8),
    suspiciousRuns: buildSuspiciousRuns(runs),
    sweepDeltas: buildSweepDeltas(runs),
    parameterOptions,
    selectedParameter: parameterFallback,
    parameterImpact: getParameterImpact(parameterFallback),
    speedQualityPoints: runs.filter((run) => run.bestAvailableMetricValue != null && (run.finalTokS ?? run.avgTokS) != null),
    preferredMetricName: pickPreferredMetricName(runs),
  };
}

export function getDashboardMeta() {
  const runs = getAllRunSummaries();
  const groups = getGroupSummaries();
  return {
    databasePath: getDatabasePath(),
    available: getDb() != null,
    runs,
    groups,
  };
}

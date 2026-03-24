export type MetricPhase = "roundtrip" | "val" | "train";

export type BestMetricName = "roundtrip_val_bpb" | "val_bpb" | "train_loss";

export type ParamValue = string | number | boolean | null;

export interface RunSummary {
  runId: string;
  scriptName: string;
  backend: string;
  status: string;
  startedAtUtc: string;
  finishedAtUtc: string | null;
  logPath: string | null;
  notes: string | null;
  experimentGroup: string | null;
  experimentLabel: string | null;
  experimentComment: string | null;
  modelParamCount: number | null;
  modelDim: number | null;
  numLayers: number | null;
  numHeads: number | null;
  numKvHeads: number | null;
  mlpMult: number | null;
  ropeBase: number | null;
  tiedEmbedLr: number | null;
  matrixLr: number | null;
  seed: number | null;
  finalTrainLoss: number | null;
  finalTrainTimeMs: number | null;
  finalTokS: number | null;
  avgTokS: number | null;
  finalValLoss: number | null;
  finalValBpb: number | null;
  finalRoundtripValLoss: number | null;
  finalRoundtripValBpb: number | null;
  maxStep: number | null;
  artifactBytes: number | null;
  artifactName: string | null;
  bestAvailableMetricName: BestMetricName | null;
  bestAvailableMetricValue: number | null;
  bestAvailableMetricPhase: MetricPhase | null;
  proxyOnly: boolean;
  trusted: boolean;
  runDurationMs: number | null;
}

export interface MetricSeriesPoint {
  step: number;
  value: number;
}

export interface MetricSeries {
  key: string;
  phase: string;
  name: string;
  points: MetricSeriesPoint[];
}

export interface RunParameter {
  name: string;
  value: ParamValue;
  valueType: string;
}

export interface RunEvent {
  id: number;
  step: number | null;
  name: string;
  message: string | null;
  recordedAtUtc: string;
}

export interface RunArtifact {
  id: number;
  name: string;
  bytes: number;
  metadataJson: string | null;
  recordedAtUtc: string;
}

export interface RunDetail {
  summary: RunSummary;
  params: RunParameter[];
  metrics: MetricSeries[];
  events: RunEvent[];
  artifacts: RunArtifact[];
}

export interface GroupSummary {
  name: string;
  runCount: number;
  latestRunId: string;
  latestStartedAtUtc: string;
  baselinePresent: boolean;
  bestRun: RunSummary;
}

export interface GroupDeltaItem {
  runId: string;
  label: string;
  comment: string | null;
  metricName: BestMetricName;
  metricValue: number;
  baselineValue: number;
  delta: number;
  trusted: boolean;
  proxyOnly: boolean;
}

export interface GroupDetail {
  summary: GroupSummary;
  runs: RunSummary[];
  baseline: RunSummary | null;
  changedParams: string[];
  deltas: GroupDeltaItem[];
  trainCurves: Array<{
    runId: string;
    label: string;
    points: MetricSeriesPoint[];
  }>;
}

export interface ParameterOption {
  name: string;
  distinctValueCount: number;
  numeric: boolean;
  runCount: number;
}

export interface ParameterImpactPoint {
  runId: string;
  label: string;
  group: string | null;
  value: ParamValue;
  numericValue: number | null;
  metricName: BestMetricName;
  metricPhase: MetricPhase;
  metricValue: number;
  trusted: boolean;
  proxyOnly: boolean;
  sampleCount: number;
}

export interface OverviewData {
  dbPath: string;
  runCount: number;
  groupCount: number;
  distinctKnobCount: number;
  bestRoundtripRun: RunSummary | null;
  bestValidationRun: RunSummary | null;
  bestProxyRun: RunSummary | null;
  bestRecentRuns: RunSummary[];
  suspiciousRuns: Array<{
    run: RunSummary;
    reason: string;
  }>;
  sweepDeltas: Array<{
    group: string;
    label: string;
    metricName: BestMetricName;
    metricValue: number;
    baselineValue: number;
    delta: number;
    proxyOnly: boolean;
    trusted: boolean;
  }>;
  parameterOptions: ParameterOption[];
  selectedParameter: string | null;
  parameterImpact: ParameterImpactPoint[];
  speedQualityPoints: RunSummary[];
  preferredMetricName: BestMetricName | null;
}

export interface RunFilters {
  backend?: string;
  status?: string;
  group?: string;
  phase?: string;
  query?: string;
  from?: string;
  to?: string;
  sort?: string;
  dir?: "asc" | "desc";
}

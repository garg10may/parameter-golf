"use client";

import Link from "next/link";
import { startTransition, useState, type FormEvent, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { cn, formatInteger, toTitleCase } from "@/lib/format";
import type { TrainingLaunchResult, TrainingSystemInfo } from "@/lib/types";

type PresetKey = "fast_dev" | "short_compare" | "full_eval";
type TrainerMode = "baseline" | "advanced";

type LaunchFormState = {
  preset: PresetKey;
  trainerMode: TrainerMode;
  runId: string;
  experimentGroup: string;
  experimentLabel: string;
  experimentComment: string;
  nprocPerNode: string;
  dataPath: string;
  tokenizerPath: string;
  vocabSize: string;
  fastDevRun: boolean;
  periodicValidation: boolean;
  valLossEvery: string;
  valBatchSize: string;
  iterations: string;
  trainBatchTokens: string;
  trainSeqLen: string;
  warmupSteps: string;
  maxWallclockSeconds: string;
  trainLogEvery: string;
  numLayers: string;
  modelDim: string;
  numHeads: string;
  numKvHeads: string;
  mlpMult: string;
  tiedEmbedLr: string;
  matrixLr: string;
  scalarLr: string;
  ropeBase: string;
  qkGainInit: string;
  gradClipNorm: string;
  useSmeargate: boolean;
  useBigramHash: boolean;
  bigramVocabSize: string;
  bigramDim: string;
  evalMode: "standard" | "sliding";
  evalStride: string;
  evalBatchSeqs: string;
  initMode: "default" | "orthogonal";
  qatEnabled: boolean;
  qatBits: string;
  swaEnabled: boolean;
  swaStartFrac: string;
  swaEvery: string;
  exportCodec: "zlib" | "zstd";
  exportCodecLevel: string;
  exportTiedEmbedMode: "int8" | "fp16";
  exportAttnWeightBits: string;
  exportMlpWeightBits: string;
  exportLateKMode: "quantized" | "fp16";
  muonWeightDecay: string;
  adamWeightDecay: string;
};

function buildPreset(preset: PresetKey): LaunchFormState {
  const base: LaunchFormState = {
    preset,
    trainerMode: "baseline",
    runId: "",
    experimentGroup: "",
    experimentLabel: "",
    experimentComment: "",
    nprocPerNode: "",
    dataPath: "",
    tokenizerPath: "",
    vocabSize: "1024",
    fastDevRun: false,
    periodicValidation: true,
    valLossEvery: "1000",
    valBatchSize: "524288",
    iterations: "20000",
    trainBatchTokens: "524288",
    trainSeqLen: "1024",
    warmupSteps: "20",
    maxWallclockSeconds: "600",
    trainLogEvery: "200",
    numLayers: "9",
    modelDim: "512",
    numHeads: "8",
    numKvHeads: "4",
    mlpMult: "2",
    tiedEmbedLr: "0.05",
    matrixLr: "0.04",
    scalarLr: "0.04",
    ropeBase: "10000",
    qkGainInit: "1.5",
    gradClipNorm: "0",
    useSmeargate: false,
    useBigramHash: false,
    bigramVocabSize: "4096",
    bigramDim: "128",
    evalMode: "standard",
    evalStride: "64",
    evalBatchSeqs: "32",
    initMode: "default",
    qatEnabled: false,
    qatBits: "6",
    swaEnabled: false,
    swaStartFrac: "0.5",
    swaEvery: "50",
    exportCodec: "zlib",
    exportCodecLevel: "0",
    exportTiedEmbedMode: "int8",
    exportAttnWeightBits: "8",
    exportMlpWeightBits: "8",
    exportLateKMode: "quantized",
    muonWeightDecay: "0",
    adamWeightDecay: "0",
  };

  if (preset === "fast_dev") {
    return {
      ...base,
      fastDevRun: true,
      periodicValidation: false,
      iterations: "100",
      trainBatchTokens: "65536",
      warmupSteps: "5",
      trainLogEvery: "10",
    };
  }
  if (preset === "short_compare") {
    return {
      ...base,
      fastDevRun: true,
      periodicValidation: true,
      valLossEvery: "100",
      iterations: "300",
      trainBatchTokens: "262144",
      warmupSteps: "10",
      trainLogEvery: "25",
    };
  }
  return base;
}

type FieldProps = {
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
};

function Field({ label, hint, children, className }: FieldProps) {
  return (
    <label className={cn("grid min-w-0 gap-1", className)}>
      <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-stone-500">{label}</span>
      {children}
      {hint ? <span className="text-xs text-stone-500">{hint}</span> : null}
    </label>
  );
}

function inputClassName() {
  return "w-full min-w-0 rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800 outline-none placeholder:text-stone-400 disabled:cursor-not-allowed disabled:bg-stone-100 disabled:text-stone-400";
}

function toggleClassName(disabled: boolean) {
  return cn(
    "flex items-center gap-3 rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-800",
    disabled && "cursor-not-allowed border-stone-200 bg-stone-100/80 text-stone-400",
  );
}

type NumericInputProps = {
  value: string;
  onChange: (value: string) => void;
  min?: number;
  max?: number;
  step?: number | "any";
  disabled?: boolean;
  required?: boolean;
};

function NumericInput({ value, onChange, min, max, step = 1, disabled, required }: NumericInputProps) {
  return (
    <input
      type="number"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className={inputClassName()}
      inputMode={step === "any" || typeof step !== "number" || !Number.isInteger(step) ? "decimal" : "numeric"}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      required={required}
    />
  );
}

export function TrainingLauncher({
  initialSystemInfo,
  initialError,
}: {
  initialSystemInfo: TrainingSystemInfo | null;
  initialError: string | null;
}) {
  const router = useRouter();
  const [form, setForm] = useState<LaunchFormState>(buildPreset("short_compare"));
  const [systemInfo, setSystemInfo] = useState<TrainingSystemInfo | null>(initialSystemInfo);
  const [systemError, setSystemError] = useState<string | null>(initialError);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [launchResult, setLaunchResult] = useState<TrainingLaunchResult | null>(null);
  const [isRefreshingSystem, setIsRefreshingSystem] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function refreshSystemInfo() {
    setIsRefreshingSystem(true);
    setSystemError(null);
    try {
      const response = await fetch("/api/training/system", { cache: "no-store" });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "Failed to refresh training system info.");
      }
      setSystemInfo(data as TrainingSystemInfo);
    } catch (error) {
      setSystemError(error instanceof Error ? error.message : "Failed to refresh training system info.");
    } finally {
      setIsRefreshingSystem(false);
    }
  }

  function updateField<K extends keyof LaunchFormState>(key: K, value: LaunchFormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function buildPayload() {
    const env: Record<string, string | number | boolean> = {
      VOCAB_SIZE: Number(form.vocabSize),
      FAST_DEV_RUN: form.fastDevRun,
      VAL_LOSS_EVERY: form.periodicValidation ? Number(form.valLossEvery) : 0,
      VAL_BATCH_SIZE: Number(form.valBatchSize),
      ITERATIONS: Number(form.iterations),
      TRAIN_BATCH_TOKENS: Number(form.trainBatchTokens),
      TRAIN_SEQ_LEN: Number(form.trainSeqLen),
      WARMUP_STEPS: Number(form.warmupSteps),
      MAX_WALLCLOCK_SECONDS: Number(form.maxWallclockSeconds),
      TRAIN_LOG_EVERY: Number(form.trainLogEvery),
      NUM_LAYERS: Number(form.numLayers),
      MODEL_DIM: Number(form.modelDim),
      NUM_HEADS: Number(form.numHeads),
      NUM_KV_HEADS: Number(form.numKvHeads),
      MLP_MULT: Number(form.mlpMult),
      TIED_EMBED_LR: Number(form.tiedEmbedLr),
      MATRIX_LR: Number(form.matrixLr),
      SCALAR_LR: Number(form.scalarLr),
      ROPE_BASE: Number(form.ropeBase),
      QK_GAIN_INIT: Number(form.qkGainInit),
      GRAD_CLIP_NORM: Number(form.gradClipNorm),
    };
    if (form.dataPath.trim()) {
      env.DATA_PATH = form.dataPath.trim();
    }
    if (form.tokenizerPath.trim()) {
      env.TOKENIZER_PATH = form.tokenizerPath.trim();
    }
    if (form.trainerMode === "advanced") {
      env.USE_SMEARGATE = form.useSmeargate;
      env.USE_BIGRAM_HASH = form.useBigramHash;
      env.EVAL_MODE = form.evalMode;
      env.INIT_MODE = form.initMode;
      env.QAT_ENABLED = form.qatEnabled;
      env.SWA_ENABLED = form.swaEnabled;
      env.EXPORT_CODEC = form.exportCodec;
      env.EXPORT_CODEC_LEVEL = Number(form.exportCodecLevel);
      env.EXPORT_TIED_EMBED_MODE = form.exportTiedEmbedMode;
      env.EXPORT_ATTN_WEIGHT_BITS = Number(form.exportAttnWeightBits);
      env.EXPORT_MLP_WEIGHT_BITS = Number(form.exportMlpWeightBits);
      env.EXPORT_LATE_K_MODE = form.exportLateKMode;
      env.MUON_WEIGHT_DECAY = Number(form.muonWeightDecay);
      env.ADAM_WEIGHT_DECAY = Number(form.adamWeightDecay);
      if (form.useBigramHash) {
        env.BIGRAM_VOCAB_SIZE = Number(form.bigramVocabSize);
        env.BIGRAM_DIM = Number(form.bigramDim);
      }
      if (form.evalMode === "sliding") {
        env.EVAL_STRIDE = Number(form.evalStride);
        env.EVAL_BATCH_SEQS = Number(form.evalBatchSeqs);
      }
      if (form.qatEnabled) {
        env.QAT_BITS = Number(form.qatBits);
      }
      if (form.swaEnabled) {
        env.SWA_START_FRAC = Number(form.swaStartFrac);
        env.SWA_EVERY = Number(form.swaEvery);
      }
    }
    return {
      trainerMode: form.trainerMode,
      runId: form.runId.trim() || undefined,
      experimentGroup: form.experimentGroup.trim() || undefined,
      experimentLabel: form.experimentLabel.trim() || undefined,
      experimentComment: form.experimentComment.trim() || undefined,
      nprocPerNode: form.nprocPerNode.trim() ? Number(form.nprocPerNode) : undefined,
      env,
    };
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setLaunchError(null);
    setLaunchResult(null);
    try {
      const response = await fetch("/api/training/launch", {
        method: "POST",
        headers: {
          "content-type": "application/json",
        },
        body: JSON.stringify(buildPayload()),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "Failed to launch training run.");
      }
      setLaunchResult(data as TrainingLaunchResult);
      startTransition(() => {
        router.refresh();
      });
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "Failed to launch training run.");
    } finally {
      setIsSubmitting(false);
    }
  }

  const advancedMode = form.trainerMode === "advanced";
  const advancedSupported = systemInfo?.supportedTrainerModes.includes("advanced") ?? false;
  const launchSupported = systemInfo != null && systemInfo.supportedTrainerModes.includes(form.trainerMode);
  const bigramControlsEnabled = advancedMode && form.useBigramHash;
  const slidingControlsEnabled = advancedMode && form.evalMode === "sliding";
  const qatControlsEnabled = advancedMode && form.qatEnabled;
  const swaControlsEnabled = advancedMode && form.swaEnabled;
  const codecLevelMax = form.exportCodec === "zstd" ? 22 : 9;
  const nprocMax = systemInfo?.cudaAvailable ? Math.max(systemInfo.cudaDeviceCount, 1) : 1;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
        <div className="min-w-0 rounded-[24px] border border-stone-200 bg-stone-50/90 p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">System</div>
              <div className="mt-1 text-lg font-semibold text-stone-950">
                {systemInfo ? `${systemInfo.platformLabel} · ${toTitleCase(systemInfo.defaultDeviceKind)}` : "Detection unavailable"}
              </div>
              <div className="mt-1 break-words text-sm text-stone-600">
                {systemInfo?.launchMessage ?? systemError ?? "The launcher could not inspect this host yet."}
              </div>
            </div>
            <button
              type="button"
              onClick={refreshSystemInfo}
              className="rounded-2xl border border-stone-300 bg-white px-3 py-2 text-sm text-stone-700 transition hover:border-stone-400"
              disabled={isRefreshingSystem}
            >
              {isRefreshingSystem ? "Refreshing..." : "Refresh system"}
            </button>
          </div>
          {systemInfo ? (
            <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <div className="min-w-0 rounded-2xl border border-stone-200 bg-white p-3">
                <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Host</div>
                <div className="mt-1 break-words text-sm text-stone-900">{systemInfo.hostname}</div>
                <div className="mt-1 font-mono text-[11px] text-stone-500">{systemInfo.machine}</div>
              </div>
              <div className="min-w-0 rounded-2xl border border-stone-200 bg-white p-3">
                <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Python</div>
                <div className="mt-1 text-sm text-stone-900">{systemInfo.pythonVersion}</div>
                <div className="mt-1 break-all font-mono text-[11px] text-stone-500">{systemInfo.pythonExecutable}</div>
              </div>
              <div className="min-w-0 rounded-2xl border border-stone-200 bg-white p-3">
                <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">GPUs</div>
                <div className="mt-1 text-sm text-stone-900">{formatInteger(systemInfo.cudaDeviceCount)}</div>
                <div className="mt-1 font-mono text-[11px] text-stone-500">default nproc={formatInteger(systemInfo.defaultNprocPerNode)}</div>
              </div>
              <div className="min-w-0 rounded-2xl border border-stone-200 bg-white p-3">
                <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Modes</div>
                <div className="mt-1 text-sm text-stone-900">
                  {systemInfo.supportedTrainerModes.length > 0 ? systemInfo.supportedTrainerModes.map(toTitleCase).join(" / ") : "None"}
                </div>
                <div className="mt-1 break-all font-mono text-[11px] text-stone-500">{systemInfo.dbPath}</div>
              </div>
            </div>
          ) : null}
          {systemInfo?.cudaDeviceNames?.length ? (
            <div className="mt-3 rounded-2xl border border-stone-200 bg-white p-3">
              <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">CUDA devices</div>
              <div className="mt-2 flex flex-wrap gap-2">
                {systemInfo.cudaDeviceNames.map((name, index) => (
                  <span key={`${name}-${index}`} className="rounded-full border border-stone-200 bg-stone-50 px-3 py-1 font-mono text-[11px] text-stone-700">
                    {index}: {name}
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>

        <div className="min-w-0 space-y-3 rounded-[24px] border border-stone-200 bg-stone-50/90 p-4">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">Launcher contract</div>
            <div className="mt-1 break-words text-sm leading-6 text-stone-700">
              The UI writes a launch request, the Python entrypoint resolves the right trainer for this machine, and the run appears in the normal SQLite-backed views as soon as tracking starts.
            </div>
          </div>
          {advancedMode && !advancedSupported ? (
            <div className="rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
              Advanced mode needs CUDA. This host currently reports {systemInfo?.launchSupport ?? "no supported accelerator"}.
            </div>
          ) : null}
          {!launchSupported ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-900">
              The selected trainer mode is not launchable on this host.
            </div>
          ) : null}
          {launchResult ? (
            <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-3 py-3 text-sm text-emerald-900">
              <div className="font-semibold">{launchResult.message}</div>
              <div className="mt-1 font-mono text-[12px]">run_id={launchResult.runId}</div>
              <div className="mt-1 font-mono text-[12px]">launch_id={launchResult.launchId}</div>
              <div className="mt-2 flex flex-wrap gap-3">
                <Link href={`/runs/${encodeURIComponent(launchResult.runId)}`} className="underline decoration-emerald-300 underline-offset-4">
                  Open run
                </Link>
                <span className="font-mono text-[12px]">{launchResult.resolvedScriptName}</span>
              </div>
            </div>
          ) : null}
          {launchError ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-3 py-3 text-sm text-rose-900">
              {launchError}
            </div>
          ) : null}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid gap-4 lg:grid-cols-4">
          <Field label="Preset">
            <select
              value={form.preset}
              onChange={(event) => setForm(buildPreset(event.target.value as PresetKey))}
              className={inputClassName()}
            >
              <option value="fast_dev">Fast dev</option>
              <option value="short_compare">Short compare</option>
              <option value="full_eval">Full eval</option>
            </select>
          </Field>
          <Field label="Trainer mode" hint="Baseline auto-selects CUDA or MLX depending on the host.">
            <select
              value={form.trainerMode}
              onChange={(event) => updateField("trainerMode", event.target.value as TrainerMode)}
              className={inputClassName()}
            >
              <option value="baseline" disabled={systemInfo != null && !systemInfo.supportedTrainerModes.includes("baseline")}>
                Baseline
              </option>
              <option value="advanced" disabled={!advancedSupported}>
                Advanced CUDA
              </option>
            </select>
          </Field>
          <Field label="Run ID" hint="Leave blank to auto-generate one.">
            <input value={form.runId} onChange={(event) => updateField("runId", event.target.value)} className={inputClassName()} maxLength={96} />
          </Field>
          <Field label="Nproc override" hint="Blank uses all detected GPUs on CUDA hosts.">
            <NumericInput value={form.nprocPerNode} onChange={(value) => updateField("nprocPerNode", value)} min={1} max={nprocMax} />
          </Field>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <Field label="Group">
            <input value={form.experimentGroup} onChange={(event) => updateField("experimentGroup", event.target.value)} className={inputClassName()} placeholder="ui_advanced_trials" maxLength={96} />
          </Field>
          <Field label="Label">
            <input value={form.experimentLabel} onChange={(event) => updateField("experimentLabel", event.target.value)} className={inputClassName()} placeholder="baseline" maxLength={96} />
          </Field>
          <Field label="Comment">
            <input value={form.experimentComment} onChange={(event) => updateField("experimentComment", event.target.value)} className={inputClassName()} placeholder="Why this run exists" maxLength={240} />
          </Field>
        </div>

        <div className="grid gap-4 lg:grid-cols-3">
          <Field label="Data path" hint="Blank uses the repo default dataset path.">
            <input value={form.dataPath} onChange={(event) => updateField("dataPath", event.target.value)} className={inputClassName()} spellCheck={false} />
          </Field>
          <Field label="Tokenizer path" hint="Blank uses the repo default tokenizer path.">
            <input value={form.tokenizerPath} onChange={(event) => updateField("tokenizerPath", event.target.value)} className={inputClassName()} spellCheck={false} />
          </Field>
          <Field label="Vocab size">
            <NumericInput value={form.vocabSize} onChange={(value) => updateField("vocabSize", value)} min={1} max={262144} required />
          </Field>
        </div>

        <div className="rounded-[24px] border border-stone-200 bg-stone-50/90 p-4">
          <div className="mb-4 text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">Execution</div>
          <div className="grid gap-4 lg:grid-cols-4">
            <Field label="Iterations">
              <NumericInput value={form.iterations} onChange={(value) => updateField("iterations", value)} min={1} max={1000000} required />
            </Field>
            <Field label="Batch tokens">
              <NumericInput value={form.trainBatchTokens} onChange={(value) => updateField("trainBatchTokens", value)} min={1} max={1000000000} required />
            </Field>
            <Field label="Train seq len">
              <NumericInput value={form.trainSeqLen} onChange={(value) => updateField("trainSeqLen", value)} min={1} max={65536} required />
            </Field>
            <Field label="Warmup steps">
              <NumericInput value={form.warmupSteps} onChange={(value) => updateField("warmupSteps", value)} min={0} max={1000000} required />
            </Field>
            <Field label="Max wallclock (s)">
              <NumericInput value={form.maxWallclockSeconds} onChange={(value) => updateField("maxWallclockSeconds", value)} min={1} max={604800} required />
            </Field>
            <Field label="Train log every">
              <NumericInput value={form.trainLogEvery} onChange={(value) => updateField("trainLogEvery", value)} min={1} max={1000000} required />
            </Field>
            <Field label="Val batch size">
              <NumericInput value={form.valBatchSize} onChange={(value) => updateField("valBatchSize", value)} min={1} max={1000000000} required />
            </Field>
            <div className="grid gap-3">
              <label className={toggleClassName(false)}>
                <input type="checkbox" checked={form.fastDevRun} onChange={(event) => updateField("fastDevRun", event.target.checked)} />
                Fast dev run
              </label>
              <label className={toggleClassName(false)}>
                <input type="checkbox" checked={form.periodicValidation} onChange={(event) => updateField("periodicValidation", event.target.checked)} />
                Periodic validation
              </label>
            </div>
          </div>
          <div className="mt-4 grid gap-4 lg:grid-cols-4">
            <Field label="Val loss every" hint="Ignored when periodic validation is off.">
              <NumericInput value={form.valLossEvery} onChange={(value) => updateField("valLossEvery", value)} min={1} max={1000000} disabled={!form.periodicValidation} required={form.periodicValidation} />
            </Field>
          </div>
        </div>

        <div className="rounded-[24px] border border-stone-200 bg-stone-50/90 p-4">
          <div className="mb-4 text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">Model and optimizer</div>
          <div className="grid gap-4 lg:grid-cols-4">
            <Field label="Layers"><NumericInput value={form.numLayers} onChange={(value) => updateField("numLayers", value)} min={1} max={256} required /></Field>
            <Field label="Model dim"><NumericInput value={form.modelDim} onChange={(value) => updateField("modelDim", value)} min={1} max={32768} required /></Field>
            <Field label="Heads"><NumericInput value={form.numHeads} onChange={(value) => updateField("numHeads", value)} min={1} max={256} required /></Field>
            <Field label="KV heads"><NumericInput value={form.numKvHeads} onChange={(value) => updateField("numKvHeads", value)} min={1} max={256} required /></Field>
            <Field label="MLP mult"><NumericInput value={form.mlpMult} onChange={(value) => updateField("mlpMult", value)} min={1} max={16} required /></Field>
            <Field label="Tied embed LR"><NumericInput value={form.tiedEmbedLr} onChange={(value) => updateField("tiedEmbedLr", value)} min={0} max={10} step="any" required /></Field>
            <Field label="Matrix LR"><NumericInput value={form.matrixLr} onChange={(value) => updateField("matrixLr", value)} min={0} max={10} step="any" required /></Field>
            <Field label="Scalar LR"><NumericInput value={form.scalarLr} onChange={(value) => updateField("scalarLr", value)} min={0} max={10} step="any" required /></Field>
            <Field label="RoPE base"><NumericInput value={form.ropeBase} onChange={(value) => updateField("ropeBase", value)} min={1} max={1000000000} step="any" required /></Field>
            <Field label="QK gain init"><NumericInput value={form.qkGainInit} onChange={(value) => updateField("qkGainInit", value)} min={0} max={1000} step="any" required /></Field>
            <Field label="Grad clip norm"><NumericInput value={form.gradClipNorm} onChange={(value) => updateField("gradClipNorm", value)} min={0} max={1000} step="any" required /></Field>
          </div>
        </div>

        <div className={cn("rounded-[24px] border p-4", advancedMode ? "border-amber-200 bg-amber-50/60" : "border-stone-200 bg-stone-50/70")}>
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">Advanced blocks</div>
              <div className="mt-1 text-sm text-stone-600">
                These env vars are only sent when trainer mode is set to Advanced CUDA.
              </div>
            </div>
            <div className="rounded-full border border-stone-300 bg-white/90 px-3 py-1 font-mono text-[11px] text-stone-500">
              {advancedMode ? "enabled" : "inactive"}
            </div>
          </div>
          <div className="grid gap-4 lg:grid-cols-4">
            <div className="grid gap-3">
              <label className={toggleClassName(!advancedMode)}>
                <input type="checkbox" checked={form.useSmeargate} onChange={(event) => updateField("useSmeargate", event.target.checked)} disabled={!advancedMode} />
                Use SmearGate
              </label>
              <label className={toggleClassName(!advancedMode)}>
                <input type="checkbox" checked={form.useBigramHash} onChange={(event) => updateField("useBigramHash", event.target.checked)} disabled={!advancedMode} />
                Use BigramHash
              </label>
              <label className={toggleClassName(!advancedMode)}>
                <input type="checkbox" checked={form.qatEnabled} onChange={(event) => updateField("qatEnabled", event.target.checked)} disabled={!advancedMode} />
                Enable QAT
              </label>
              <label className={toggleClassName(!advancedMode)}>
                <input type="checkbox" checked={form.swaEnabled} onChange={(event) => updateField("swaEnabled", event.target.checked)} disabled={!advancedMode} />
                Enable SWA
              </label>
            </div>
            <Field label="Eval mode">
              <select value={form.evalMode} onChange={(event) => updateField("evalMode", event.target.value as LaunchFormState["evalMode"])} className={inputClassName()} disabled={!advancedMode}>
                <option value="standard">Standard</option>
                <option value="sliding">Sliding</option>
              </select>
            </Field>
            <Field label="Init mode">
              <select value={form.initMode} onChange={(event) => updateField("initMode", event.target.value as LaunchFormState["initMode"])} className={inputClassName()} disabled={!advancedMode}>
                <option value="default">Default</option>
                <option value="orthogonal">Orthogonal</option>
              </select>
            </Field>
            <Field label="Export codec">
              <select value={form.exportCodec} onChange={(event) => updateField("exportCodec", event.target.value as LaunchFormState["exportCodec"])} className={inputClassName()} disabled={!advancedMode}>
                <option value="zlib">zlib</option>
                <option value="zstd">zstd</option>
              </select>
            </Field>
            <Field label="Bigram vocab" hint="Only used when BigramHash is enabled."><NumericInput value={form.bigramVocabSize} onChange={(value) => updateField("bigramVocabSize", value)} min={1} max={1048576} disabled={!bigramControlsEnabled} required={bigramControlsEnabled} /></Field>
            <Field label="Bigram dim" hint="Only used when BigramHash is enabled."><NumericInput value={form.bigramDim} onChange={(value) => updateField("bigramDim", value)} min={1} max={8192} disabled={!bigramControlsEnabled} required={bigramControlsEnabled} /></Field>
            <Field label="Eval stride" hint="Only used in sliding eval mode."><NumericInput value={form.evalStride} onChange={(value) => updateField("evalStride", value)} min={1} max={65536} disabled={!slidingControlsEnabled} required={slidingControlsEnabled} /></Field>
            <Field label="Eval batch seqs" hint="Only used in sliding eval mode."><NumericInput value={form.evalBatchSeqs} onChange={(value) => updateField("evalBatchSeqs", value)} min={1} max={8192} disabled={!slidingControlsEnabled} required={slidingControlsEnabled} /></Field>
            <Field label="QAT bits" hint="Only used when QAT is enabled.">
              <select value={form.qatBits} onChange={(event) => updateField("qatBits", event.target.value)} className={inputClassName()} disabled={!qatControlsEnabled}>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
              </select>
            </Field>
            <Field label="SWA start frac" hint="Only used when SWA is enabled."><NumericInput value={form.swaStartFrac} onChange={(value) => updateField("swaStartFrac", value)} min={0} max={1} step={0.05} disabled={!swaControlsEnabled} required={swaControlsEnabled} /></Field>
            <Field label="SWA every" hint="Only used when SWA is enabled."><NumericInput value={form.swaEvery} onChange={(value) => updateField("swaEvery", value)} min={1} max={1000000} disabled={!swaControlsEnabled} required={swaControlsEnabled} /></Field>
            <Field label="Codec level"><NumericInput value={form.exportCodecLevel} onChange={(value) => updateField("exportCodecLevel", value)} min={0} max={codecLevelMax} disabled={!advancedMode} required={advancedMode} /></Field>
            <Field label="Tied embed export">
              <select value={form.exportTiedEmbedMode} onChange={(event) => updateField("exportTiedEmbedMode", event.target.value as LaunchFormState["exportTiedEmbedMode"])} className={inputClassName()} disabled={!advancedMode}>
                <option value="int8">int8</option>
                <option value="fp16">fp16</option>
              </select>
            </Field>
            <Field label="Attention bits">
              <select value={form.exportAttnWeightBits} onChange={(event) => updateField("exportAttnWeightBits", event.target.value)} className={inputClassName()} disabled={!advancedMode}>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
              </select>
            </Field>
            <Field label="MLP bits">
              <select value={form.exportMlpWeightBits} onChange={(event) => updateField("exportMlpWeightBits", event.target.value)} className={inputClassName()} disabled={!advancedMode}>
                <option value="4">4</option>
                <option value="5">5</option>
                <option value="6">6</option>
                <option value="7">7</option>
                <option value="8">8</option>
              </select>
            </Field>
            <Field label="Late-K mode">
              <select value={form.exportLateKMode} onChange={(event) => updateField("exportLateKMode", event.target.value as LaunchFormState["exportLateKMode"])} className={inputClassName()} disabled={!advancedMode}>
                <option value="quantized">Quantized</option>
                <option value="fp16">fp16</option>
              </select>
            </Field>
            <Field label="Muon WD"><NumericInput value={form.muonWeightDecay} onChange={(value) => updateField("muonWeightDecay", value)} min={0} max={10} step="any" disabled={!advancedMode} required={advancedMode} /></Field>
            <Field label="Adam WD"><NumericInput value={form.adamWeightDecay} onChange={(value) => updateField("adamWeightDecay", value)} min={0} max={10} step="any" disabled={!advancedMode} required={advancedMode} /></Field>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="text-sm text-stone-600">
            The launcher always writes a background launch request first, then the trainer populates the standard run tables.
          </div>
          <button
            type="submit"
            className="rounded-2xl bg-stone-950 px-5 py-3 text-sm font-medium text-stone-50 transition hover:bg-stone-800 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={isSubmitting || !launchSupported}
          >
            {isSubmitting ? "Launching..." : "Launch run"}
          </button>
        </div>
      </form>
    </div>
  );
}

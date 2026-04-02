import Link from "next/link";
import { notFound } from "next/navigation";
import { MetricBadge, ProxyBadge } from "@/components/badges";
import { LearningCurvesChart } from "@/components/charts";
import { RunLogPanel } from "@/components/run-log-panel";
import { SectionFrame, StatTile } from "@/components/section-frame";
import { formatBytes, formatCompactNumber, formatDuration, formatModelFootprint, formatNumber, formatRelativeDate, toTitleCase } from "@/lib/format";
import { getRunDetail } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const detail = getRunDetail(runId);
  if (!detail) {
    notFound();
  }

  const trainLossSeries = detail.metrics.find((item) => item.phase === "train" && item.name === "train_loss");
  const tokSSeries = detail.metrics.find((item) => item.phase === "train" && item.name === "tok_s");

  return (
    <div className="space-y-6">
      <SectionFrame
        eyebrow="Run detail"
        title={detail.summary.experimentLabel ?? detail.summary.runId}
        subtitle={detail.summary.experimentComment ?? detail.summary.notes ?? "No comment was recorded for this run."}
        action={
          <div className="flex flex-wrap gap-2">
            <ProxyBadge run={detail.summary} />
            <MetricBadge run={detail.summary} />
          </div>
        }
      >
        <div className="grid gap-4 lg:grid-cols-5">
          <StatTile label="Best score" value={formatNumber(detail.summary.bestAvailableMetricValue)} hint={detail.summary.bestAvailableMetricName ?? "No metric"} />
          <StatTile label="Train loss" value={formatNumber(detail.summary.finalTrainLoss)} hint={`Max step ${detail.summary.maxStep ?? "—"}`} />
          <StatTile label="Validation bpb" value={formatNumber(detail.summary.finalValBpb)} hint="Validation phase" />
          <StatTile label="Roundtrip bpb" value={formatNumber(detail.summary.finalRoundtripValBpb)} hint="Trusted phase" />
          <StatTile label="Duration" value={formatDuration(detail.summary.runDurationMs)} hint={`${formatCompactNumber(detail.summary.finalTokS ?? detail.summary.avgTokS)} tok/s`} />
        </div>
      </SectionFrame>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
        <SectionFrame eyebrow="Metadata" title="Run metadata" subtitle="Context and traceability for this run.">
          <dl className="grid gap-x-6 gap-y-3 sm:grid-cols-2">
            <MetaRow label="Run ID" value={detail.summary.runId} mono />
            <MetaRow label="Backend" value={detail.summary.backend} />
            <MetaRow label="Group" value={detail.summary.experimentGroup ?? "Ad hoc"} />
            <MetaRow label="Started" value={formatRelativeDate(detail.summary.startedAtUtc)} />
            <MetaRow label="Status" value={toTitleCase(detail.summary.status)} />
            <MetaRow label="Log path" value={detail.summary.logPath ?? "—"} mono />
            <MetaRow label="Model" value={formatModelFootprint(detail.summary)} mono />
            <MetaRow label="Artifact bytes" value={formatBytes(detail.summary.artifactBytes)} mono />
          </dl>
          {detail.summary.experimentGroup ? (
            <div className="mt-4">
              <Link
                href={`/groups/${encodeURIComponent(detail.summary.experimentGroup)}`}
                className="inline-flex items-center rounded-full border border-stone-300 bg-white px-3 py-2 text-sm font-medium text-stone-700 transition hover:border-stone-500"
              >
                Open sweep view
              </Link>
            </div>
          ) : null}
        </SectionFrame>

        <SectionFrame eyebrow="Artifacts" title="Saved outputs" subtitle="Serialized artifacts and logged events tied to this run.">
          <div className="space-y-3">
            {detail.artifacts.length > 0 ? (
              detail.artifacts.map((artifact) => (
                <div key={artifact.id} className="rounded-[24px] border border-stone-200 bg-stone-50/80 p-4">
                  <div className="font-semibold text-stone-950">{artifact.name}</div>
                  <div className="mt-1 font-mono text-sm text-stone-600">{formatBytes(artifact.bytes)}</div>
                  <div className="mt-1 text-xs text-stone-500">{formatRelativeDate(artifact.recordedAtUtc)}</div>
                </div>
              ))
            ) : (
              <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
                No artifacts were recorded for this run.
              </div>
            )}
            <div className="rounded-[24px] border border-stone-200 bg-stone-50/80 p-4">
              <div className="text-sm font-semibold text-stone-950">Events</div>
              <div className="mt-3 space-y-3">
                {detail.events.length > 0 ? (
                  detail.events.map((event) => (
                    <div key={event.id} className="border-t border-stone-200/80 pt-3 first:border-t-0 first:pt-0">
                      <div className="flex items-center justify-between gap-3">
                        <div className="font-medium text-stone-800">{event.name}</div>
                        <div className="font-mono text-[11px] text-stone-500">{event.step != null ? `step ${event.step}` : formatRelativeDate(event.recordedAtUtc)}</div>
                      </div>
                      {event.message ? <div className="mt-1 text-sm text-stone-600">{event.message}</div> : null}
                    </div>
                  ))
                ) : (
                  <div className="text-sm text-stone-600">No run events were recorded.</div>
                )}
              </div>
            </div>
          </div>
        </SectionFrame>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <SectionFrame eyebrow="Learning curve" title="Train loss over steps" subtitle="Use this to compare optimization shape, not just the final point.">
          {trainLossSeries ? (
            <LearningCurvesChart series={[{ label: "train_loss", points: trainLossSeries.points }]} yAxisName="train_loss" />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
              No train-loss series was logged for this run.
            </div>
          )}
        </SectionFrame>
        <SectionFrame eyebrow="Throughput" title="Tokens per second" subtitle="Proxy speed matters, but speed without validation can be misleading.">
          {tokSSeries ? (
            <LearningCurvesChart series={[{ label: "tok_s", points: tokSSeries.points }]} yAxisName="tok/s" />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
              No throughput series was logged for this run.
            </div>
          )}
        </SectionFrame>
      </div>

      <SectionFrame
        eyebrow="Live logs"
        title="Runtime output"
        subtitle="Tail the background process output live from the dashboard. Console log matches the spawned process stream when available."
      >
        <RunLogPanel runId={detail.summary.runId} initialStatus={detail.summary.status} />
      </SectionFrame>

      <SectionFrame eyebrow="Parameters" title="Recorded hyperparameters" subtitle="Everything stored in SQLite for this run, sorted alphabetically.">
        <div className="overflow-hidden rounded-[24px] border border-stone-200">
          <div className="grid max-h-[620px] overflow-auto">
            <table className="min-w-full border-collapse text-left text-sm">
              <thead className="bg-stone-100/90 text-stone-600">
                <tr>
                  <th className="px-4 py-3 font-medium">Name</th>
                  <th className="px-4 py-3 font-medium">Type</th>
                  <th className="px-4 py-3 font-medium">Value</th>
                </tr>
              </thead>
              <tbody>
                {detail.params.map((param) => (
                  <tr key={param.name} className="border-t border-stone-200/80 bg-white/70">
                    <td className="px-4 py-3 font-mono text-[12px] text-stone-700">{param.name}</td>
                    <td className="px-4 py-3 text-stone-600">{param.valueType}</td>
                    <td className="px-4 py-3 font-mono text-[12px] text-stone-900">{String(param.value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </SectionFrame>
    </div>
  );
}

function MetaRow({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">{label}</dt>
      <dd className={`mt-1 min-w-0 break-all text-sm leading-6 text-stone-900 ${mono ? "font-mono" : ""}`}>{value}</dd>
    </div>
  );
}

import Link from "next/link";
import { notFound } from "next/navigation";
import { MetricBadge, ProxyBadge } from "@/components/badges";
import { LearningCurvesChart, SweepDeltaChart } from "@/components/charts";
import { SectionFrame, StatTile } from "@/components/section-frame";
import { formatCompactNumber, formatModelFootprint, formatNumber, formatRelativeDate, toTitleCase } from "@/lib/format";
import { getGroupDetail } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default async function GroupDetailPage({
  params,
}: {
  params: Promise<{ group: string }>;
}) {
  const { group } = await params;
  const detail = getGroupDetail(group);
  if (!detail) {
    notFound();
  }

  return (
    <div className="space-y-6">
      <SectionFrame
        eyebrow="Sweep detail"
        title={detail.summary.name}
        subtitle={`Best run: ${detail.summary.bestRun.experimentLabel ?? detail.summary.bestRun.runId}. Baseline present: ${detail.summary.baselinePresent ? "yes" : "no"}.`}
      >
        <div className="grid gap-4 lg:grid-cols-4">
          <StatTile label="Run count" value={formatCompactNumber(detail.summary.runCount)} hint={`Latest ${formatRelativeDate(detail.summary.latestStartedAtUtc)}`} />
          <StatTile label="Best score" value={formatNumber(detail.summary.bestRun.bestAvailableMetricValue)} hint={detail.summary.bestRun.experimentLabel ?? detail.summary.bestRun.runId} />
          <StatTile label="Baseline" value={detail.baseline ? detail.baseline.experimentLabel ?? detail.baseline.runId : "—"} hint={detail.baseline ? "Reference run found" : "This group has no baseline label"} />
          <StatTile label="Changed params" value={formatCompactNumber(detail.changedParams.length)} hint={detail.changedParams.join(", ") || "No varying params detected"} />
        </div>
      </SectionFrame>

      <div className="grid gap-6 xl:grid-cols-[1.15fr_1fr]">
        <SectionFrame
          eyebrow="Delta view"
          title="Baseline versus variants"
          subtitle="Each bar shows how far a variant moved relative to the sweep baseline under the preferred group metric."
        >
          {detail.deltas.length > 0 ? (
            <SweepDeltaChart
              items={detail.deltas.map((item) => ({
                group: detail.summary.name,
                label: item.label,
                metricName: item.metricName,
                delta: item.delta,
                baselineValue: item.baselineValue,
                metricValue: item.metricValue,
                proxyOnly: item.proxyOnly,
              }))}
            />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
              A baseline exists only when one run uses the `baseline` label and shares a comparable metric with the others.
            </div>
          )}
        </SectionFrame>

        <SectionFrame
          eyebrow="Train curves"
          title="Optimization comparison"
          subtitle="Useful for spotting which variant improves earlier and which one plateaus later."
        >
          {detail.trainCurves.length > 0 ? (
            <LearningCurvesChart series={detail.trainCurves} yAxisName="train_loss" />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
              No train-loss curves are available for this sweep.
            </div>
          )}
        </SectionFrame>
      </div>

      <SectionFrame
        eyebrow="Runs"
        title="Sweep members"
        subtitle="Compare comments, trust level, and final score without opening each run first."
      >
        <div className="overflow-hidden rounded-[24px] border border-stone-200">
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-left text-sm">
              <thead className="bg-stone-100/90 text-stone-600">
                <tr>
                  <th className="px-4 py-3 font-medium">Label</th>
                  <th className="px-4 py-3 font-medium">Score</th>
                  <th className="px-4 py-3 font-medium">Trust</th>
                  <th className="px-4 py-3 font-medium">Comment</th>
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">Started</th>
                </tr>
              </thead>
              <tbody>
                {detail.runs.map((run) => (
                  <tr key={run.runId} className="border-t border-stone-200/80 bg-white/70">
                    <td className="px-4 py-3 align-top">
                      <Link href={`/runs/${encodeURIComponent(run.runId)}`} className="block">
                        <div className="font-semibold text-stone-950">{run.experimentLabel ?? run.runId}</div>
                        <div className="mt-1 font-mono text-[11px] text-stone-500">{run.runId}</div>
                      </Link>
                    </td>
                    <td className="px-4 py-3 align-top font-mono text-stone-900">{formatNumber(run.bestAvailableMetricValue)}</td>
                    <td className="px-4 py-3 align-top">
                      <div className="flex flex-wrap gap-2">
                        <ProxyBadge run={run} />
                        <MetricBadge run={run} />
                      </div>
                    </td>
                    <td className="px-4 py-3 align-top text-stone-700">{run.experimentComment ?? run.notes ?? "—"}</td>
                    <td className="px-4 py-3 align-top font-mono text-stone-700">{formatModelFootprint(run)} · {toTitleCase(run.status)}</td>
                    <td className="px-4 py-3 align-top text-stone-600">{formatRelativeDate(run.startedAtUtc)}</td>
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

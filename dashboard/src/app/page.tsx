import Link from "next/link";
import { MetricBadge, ProxyBadge } from "@/components/badges";
import { ParameterImpactChart, SpeedQualityChart, SweepDeltaChart } from "@/components/charts";
import { ParameterSelectForm } from "@/components/parameter-select-form";
import { SectionFrame, StatTile } from "@/components/section-frame";
import { formatCompactNumber, formatDuration, formatModelFootprint, formatNumber, formatRelativeDate, metricDirectionHint } from "@/lib/format";
import { getOverviewData } from "@/lib/queries";

export const dynamic = "force-dynamic";

function getSearchValue(
  value: string | string[] | undefined,
) {
  if (Array.isArray(value)) return value[0];
  return value;
}

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const selectedParameter = getSearchValue(params.param) ?? null;
  const overview = getOverviewData(selectedParameter);

  return (
    <div className="space-y-6">
      <SectionFrame
        eyebrow="Overview"
        title="Signals-first experiment intelligence"
        subtitle={`Ranking prefers roundtrip val_bpb, then validation val_bpb, then train-loss proxy. Database: ${overview.dbPath}`}
      >
        <div className="grid gap-4 lg:grid-cols-5">
          <StatTile
            label="Best roundtrip"
            value={overview.bestRoundtripRun ? formatNumber(overview.bestRoundtripRun.finalRoundtripValBpb) : "—"}
            hint={overview.bestRoundtripRun?.experimentLabel ?? overview.bestRoundtripRun?.runId ?? "No trusted runs yet"}
          />
          <StatTile
            label="Best validation"
            value={overview.bestValidationRun ? formatNumber(overview.bestValidationRun.finalValBpb) : "—"}
            hint={overview.bestValidationRun?.experimentLabel ?? overview.bestValidationRun?.runId ?? "No validation-backed runs yet"}
          />
          <StatTile
            label="Best proxy"
            value={overview.bestProxyRun ? formatNumber(overview.bestProxyRun.finalTrainLoss) : "—"}
            hint={overview.bestProxyRun?.experimentLabel ?? overview.bestProxyRun?.runId ?? "No train-only runs yet"}
          />
          <StatTile label="Total runs" value={formatCompactNumber(overview.runCount)} hint={`${overview.groupCount} sweep groups`} />
          <StatTile label="Distinct knobs tried" value={formatCompactNumber(overview.distinctKnobCount)} hint={metricDirectionHint(overview.preferredMetricName)} />
        </div>
      </SectionFrame>

      <div className="grid gap-6 xl:grid-cols-[1.6fr_1fr]">
        <SectionFrame
          eyebrow="Best recent runs"
          title="Candidate board"
          subtitle="Fast way to see what currently looks strongest, with trust level called out explicitly."
        >
          <div className="space-y-3">
            {overview.bestRecentRuns.map((run, index) => (
              <Link
                key={run.runId}
                href={`/runs/${encodeURIComponent(run.runId)}`}
                className="grid gap-3 rounded-[24px] border border-stone-200 bg-stone-50/80 p-4 transition hover:border-stone-400 hover:bg-white"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-stone-500">Rank #{index + 1}</div>
                    <div className="mt-1 text-base font-semibold text-stone-950">{run.experimentLabel ?? run.runId}</div>
                    <div className="mt-1 text-sm text-stone-600">
                      {run.experimentGroup ?? "Ad hoc run"} · {formatRelativeDate(run.startedAtUtc)}
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <ProxyBadge run={run} />
                    <MetricBadge run={run} />
                  </div>
                </div>
                <div className="grid gap-3 sm:grid-cols-4">
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Score</div>
                    <div className="font-mono text-sm text-stone-900">{formatNumber(run.bestAvailableMetricValue)}</div>
                  </div>
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Model</div>
                    <div className="font-mono text-sm text-stone-900">{formatModelFootprint(run)}</div>
                  </div>
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Throughput</div>
                    <div className="font-mono text-sm text-stone-900">{formatCompactNumber(run.finalTokS ?? run.avgTokS)}</div>
                  </div>
                  <div>
                    <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Duration</div>
                    <div className="font-mono text-sm text-stone-900">{formatDuration(run.runDurationMs)}</div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </SectionFrame>

        <SectionFrame
          eyebrow="Anomalies"
          title="Runs that need skepticism"
          subtitle="These are the cases most likely to mislead ranking if you only skim raw numbers."
        >
          <div className="space-y-3">
            {overview.suspiciousRuns.length === 0 ? (
              <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-6 text-sm text-stone-600">
                No suspicious runs are currently flagged.
              </div>
            ) : (
              overview.suspiciousRuns.map(({ run, reason }) => (
                <Link
                  key={run.runId}
                  href={`/runs/${encodeURIComponent(run.runId)}`}
                  className="block rounded-[24px] border border-amber-200 bg-amber-50/80 p-4 transition hover:border-amber-400"
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-stone-950">{run.experimentLabel ?? run.runId}</div>
                    <ProxyBadge run={run} />
                  </div>
                  <p className="mt-2 text-sm leading-6 text-stone-700">{reason}</p>
                </Link>
              ))
            )}
          </div>
        </SectionFrame>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <SectionFrame
          eyebrow="Sweep deltas"
          title="Baseline versus variants"
          subtitle="Top sweep deviations ranked by absolute movement from baseline. Negative deltas are better because lower is better."
        >
          {overview.sweepDeltas.length > 0 ? (
            <SweepDeltaChart items={overview.sweepDeltas} />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-8 text-sm text-stone-600">
              No sweep baselines with comparable metrics are available yet.
            </div>
          )}
        </SectionFrame>

        <SectionFrame
          eyebrow="Parameter impact"
          title="Knob influence"
          subtitle="Computed from final per-run scores only, so the chart highlights end-state behavior instead of every logged step."
          action={
            <ParameterSelectForm options={overview.parameterOptions} selectedParameter={overview.selectedParameter} />
          }
        >
          {overview.selectedParameter && overview.parameterImpact.length > 0 ? (
            <ParameterImpactChart points={overview.parameterImpact} parameterName={overview.selectedParameter} />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-8 text-sm text-stone-600">
              There are not enough varying parameter values yet to estimate directional signal.
            </div>
          )}
        </SectionFrame>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1.2fr_1fr]">
        <SectionFrame
          eyebrow="Tradeoffs"
          title="Speed versus score"
          subtitle="Fast proxy runs are useful, but they need validation before being treated as winners."
        >
          {overview.speedQualityPoints.length > 0 ? (
            <SpeedQualityChart runs={overview.speedQualityPoints} />
          ) : (
            <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-8 text-sm text-stone-600">
              Throughput and metric points will appear here once runs are logged.
            </div>
          )}
        </SectionFrame>

        <SectionFrame
          eyebrow="Read path"
          title="How to use this screen"
          subtitle="This dashboard is optimized to reduce manual log reading once the run count gets large."
        >
          <ol className="space-y-4 text-sm leading-6 text-stone-700">
            <li>
              <span className="font-semibold text-stone-950">1.</span> Start with the top summary and candidate board to see who is leading under the ranking rule.
            </li>
            <li>
              <span className="font-semibold text-stone-950">2.</span> Use sweep deltas to understand whether a variant beat its own baseline, not just the whole pool.
            </li>
            <li>
              <span className="font-semibold text-stone-950">3.</span> Use parameter impact and speed/quality scatter to decide which knobs deserve a longer validated run.
            </li>
            <li>
              <span className="font-semibold text-stone-950">4.</span> Treat anything marked proxy-only as a screening signal, not a final ranking.
            </li>
          </ol>
        </SectionFrame>
      </div>
    </div>
  );
}

import Link from "next/link";
import { MetricBadge, ProxyBadge } from "@/components/badges";
import { SectionFrame } from "@/components/section-frame";
import { formatModelFootprint, formatNumber, formatRelativeDate } from "@/lib/format";
import { getGroupSummaries } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default function GroupsPage() {
  const groups = getGroupSummaries();

  return (
    <div className="space-y-6">
      <SectionFrame
        eyebrow="Sweeps"
        title="Experiment groups"
        subtitle="Each group clusters related runs so you can compare variants against a baseline instead of ranking everything blindly."
      >
        <div className="grid gap-4">
          {groups.map((group) => (
            <Link
              key={group.name}
              href={`/groups/${encodeURIComponent(group.name)}`}
              className="grid gap-3 rounded-[26px] border border-stone-200 bg-white/85 p-4 transition hover:border-stone-400"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="text-lg font-semibold tracking-tight text-stone-950">{group.name}</div>
                  <div className="mt-1 text-sm text-stone-600">
                    {group.runCount} runs · latest {formatRelativeDate(group.latestStartedAtUtc)}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <ProxyBadge run={group.bestRun} />
                  <MetricBadge run={group.bestRun} />
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-4">
                <div>
                  <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Best score</div>
                  <div className="font-mono text-sm text-stone-900">{formatNumber(group.bestRun.bestAvailableMetricValue)}</div>
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Best run</div>
                  <div className="text-sm text-stone-900">{group.bestRun.experimentLabel ?? group.bestRun.runId}</div>
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Baseline</div>
                  <div className="text-sm text-stone-900">{group.baselinePresent ? "Present" : "Missing"}</div>
                </div>
                <div>
                  <div className="text-[11px] uppercase tracking-[0.12em] text-stone-500">Model</div>
                  <div className="font-mono text-sm text-stone-900">{formatModelFootprint(group.bestRun)}</div>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </SectionFrame>
    </div>
  );
}

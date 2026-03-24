import Link from "next/link";
import { MetricBadge, ProxyBadge } from "@/components/badges";
import { RunsFilterForm } from "@/components/runs-filter-form";
import { SectionFrame } from "@/components/section-frame";
import { formatCompactNumber, formatDuration, formatModelFootprint, formatNumber, formatRelativeDate } from "@/lib/format";
import { getGroupSummaries, getRunSummaries } from "@/lib/queries";

export const dynamic = "force-dynamic";

function getSearchValue(value: string | string[] | undefined) {
  return Array.isArray(value) ? value[0] : value;
}

export default async function RunsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const defaults = {
    q: getSearchValue(params.q) ?? "",
    backend: getSearchValue(params.backend) ?? "all",
    status: getSearchValue(params.status) ?? "all",
    group: getSearchValue(params.group) ?? "all",
    phase: getSearchValue(params.phase) ?? "all",
    from: getSearchValue(params.from) ?? "",
    to: getSearchValue(params.to) ?? "",
    sort: getSearchValue(params.sort) ?? "started",
    dir: getSearchValue(params.dir) ?? "desc",
  };
  const runs = getRunSummaries({
    backend: defaults.backend,
    status: defaults.status,
    group: defaults.group,
    phase: defaults.phase,
    query: defaults.q,
    from: defaults.from,
    to: defaults.to,
    sort: defaults.sort,
    dir: defaults.dir as "asc" | "desc",
  });
  const groups = getGroupSummaries().map((group) => group.name);

  return (
    <div className="space-y-6">
      <SectionFrame
        eyebrow="Runs"
        title="Filterable run ledger"
        subtitle="Dense but readable: every run, sortable by score, duration, and model size."
      >
        <RunsFilterForm groups={groups} defaults={defaults} />
      </SectionFrame>

      <SectionFrame
        eyebrow="Results"
        title={`${runs.length} runs in view`}
        subtitle="Trusted runs are validation-backed. Proxy runs only have train-loss evidence."
      >
        <div className="overflow-hidden rounded-[24px] border border-stone-200">
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse text-left text-sm">
              <thead className="bg-stone-100/90 text-stone-600">
                <tr>
                  <th className="px-4 py-3 font-medium">Run</th>
                  <th className="px-4 py-3 font-medium">Group</th>
                  <th className="px-4 py-3 font-medium">Metric</th>
                  <th className="px-4 py-3 font-medium">Trust</th>
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">tok/s</th>
                  <th className="px-4 py-3 font-medium">Duration</th>
                  <th className="px-4 py-3 font-medium">Started</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.runId} className="border-t border-stone-200/80 bg-white/70">
                    <td className="px-4 py-3 align-top">
                      <Link href={`/runs/${encodeURIComponent(run.runId)}`} className="block">
                        <div className="font-semibold text-stone-950">{run.experimentLabel ?? run.runId}</div>
                        <div className="mt-1 font-mono text-[11px] text-stone-500">{run.runId}</div>
                      </Link>
                    </td>
                    <td className="px-4 py-3 align-top text-stone-700">{run.experimentGroup ?? "Ad hoc"}</td>
                    <td className="px-4 py-3 align-top font-mono text-stone-900">{formatNumber(run.bestAvailableMetricValue)}</td>
                    <td className="px-4 py-3 align-top">
                      <div className="flex flex-wrap gap-2">
                        <ProxyBadge run={run} />
                        <MetricBadge run={run} />
                      </div>
                    </td>
                    <td className="px-4 py-3 align-top font-mono text-stone-700">{formatModelFootprint(run)}</td>
                    <td className="px-4 py-3 align-top font-mono text-stone-700">{formatCompactNumber(run.finalTokS ?? run.avgTokS)}</td>
                    <td className="px-4 py-3 align-top font-mono text-stone-700">{formatDuration(run.runDurationMs)}</td>
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

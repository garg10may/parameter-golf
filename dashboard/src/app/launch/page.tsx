import Link from "next/link";
import { TrainingLauncher } from "@/components/training-launcher";
import { SectionFrame } from "@/components/section-frame";
import { formatInteger, formatRelativeDate, toTitleCase } from "@/lib/format";
import { getLaunchRequests } from "@/lib/queries";
import { getTrainingSystemInfo } from "@/lib/training-control";
import type { TrainingSystemInfo } from "@/lib/types";

export const dynamic = "force-dynamic";

export default function LaunchPage() {
  let systemInfo: TrainingSystemInfo | null = null;
  let systemError: string | null = null;
  try {
    systemInfo = getTrainingSystemInfo();
  } catch (error) {
    systemError = error instanceof Error ? error.message : "Failed to detect training environment.";
  }

  const launches = getLaunchRequests(20);

  return (
    <div className="space-y-6">
      <SectionFrame
        eyebrow="Launch"
        title="Run training from the dashboard"
        subtitle="Choose a preset, tune the knobs, and let the launcher resolve the right training entrypoint for this machine."
      >
        <TrainingLauncher initialSystemInfo={systemInfo} initialError={systemError} />
      </SectionFrame>

      <SectionFrame
        eyebrow="Recent launches"
        title="Background run requests"
        subtitle="Every UI launch is recorded in SQLite before the trainer starts, then linked to the tracked run once it comes online."
      >
        {launches.length === 0 ? (
          <div className="rounded-[24px] border border-dashed border-stone-300 bg-stone-50/70 p-8 text-sm text-stone-600">
            No UI launch requests have been recorded yet.
          </div>
        ) : (
          <div className="overflow-hidden rounded-[24px] border border-stone-200">
            <div className="overflow-x-auto">
              <table className="min-w-full border-collapse text-left text-sm">
                <thead className="bg-stone-100/90 text-stone-600">
                  <tr>
                    <th className="px-4 py-3 font-medium">Run</th>
                    <th className="px-4 py-3 font-medium">Launch</th>
                    <th className="px-4 py-3 font-medium">Trainer</th>
                    <th className="px-4 py-3 font-medium">System</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                    <th className="px-4 py-3 font-medium">When</th>
                  </tr>
                </thead>
                <tbody>
                  {launches.map((launch) => (
                    <tr key={launch.launchId} className="border-t border-stone-200/80 bg-white/70">
                      <td className="px-4 py-3 align-top">
                        {launch.runStatus ? (
                          <Link href={`/runs/${encodeURIComponent(launch.runId)}`} className="block">
                            <div className="font-semibold text-stone-950">{launch.experimentLabel ?? launch.runId}</div>
                            <div className="mt-1 font-mono text-[11px] text-stone-500">{launch.runId}</div>
                          </Link>
                        ) : (
                          <div>
                            <div className="font-semibold text-stone-950">{launch.experimentLabel ?? launch.runId}</div>
                            <div className="mt-1 font-mono text-[11px] text-stone-500">{launch.runId}</div>
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3 align-top">
                        <div className="font-mono text-[11px] text-stone-500">{launch.launchId}</div>
                        <div className="mt-1 text-sm text-stone-700">{launch.experimentGroup ?? "Ad hoc"}</div>
                      </td>
                      <td className="px-4 py-3 align-top text-stone-700">
                        <div>{toTitleCase(launch.trainerMode)}</div>
                        <div className="mt-1 font-mono text-[11px] text-stone-500">{launch.resolvedScriptName ?? "—"}</div>
                      </td>
                      <td className="px-4 py-3 align-top text-stone-700">
                        <div>{launch.platform ? toTitleCase(launch.platform) : "—"}</div>
                        <div className="mt-1 font-mono text-[11px] text-stone-500">
                          {launch.deviceKind ?? "—"} · {formatInteger(launch.deviceCount)}
                        </div>
                      </td>
                      <td className="px-4 py-3 align-top text-stone-700">
                        <div className="font-medium text-stone-900">{toTitleCase(launch.status)}</div>
                        <div className="mt-1 text-xs text-stone-500">{launch.runStatus ? `run=${launch.runStatus}` : launch.message ?? "—"}</div>
                      </td>
                      <td className="px-4 py-3 align-top text-stone-600">{formatRelativeDate(launch.requestedAtUtc)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </SectionFrame>
    </div>
  );
}

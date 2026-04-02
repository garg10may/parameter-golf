import { getRunLogSnapshot, type RunLogSource } from "@/lib/run-logs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function isRunLogSource(value: string | null): value is RunLogSource {
  return value === "console" || value === "run";
}

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> },
) {
  try {
    const { runId } = await context.params;
    const { searchParams } = new URL(request.url);
    const sourceParam = searchParams.get("source");
    const requestedSource = isRunLogSource(sourceParam) ? sourceParam : "console";
    const snapshot = await getRunLogSnapshot(runId, requestedSource);
    if (!snapshot) {
      return Response.json({ error: `Run ${runId} was not found.` }, { status: 404 });
    }
    return Response.json(snapshot);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to read run log.";
    return Response.json({ error: message }, { status: 500 });
  }
}

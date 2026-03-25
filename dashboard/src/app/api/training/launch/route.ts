import { launchTrainingRun } from "@/lib/training-control";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    return Response.json(launchTrainingRun(payload));
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to launch training run.";
    return Response.json({ error: message }, { status: 400 });
  }
}

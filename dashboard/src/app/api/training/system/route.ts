import { getTrainingSystemInfo } from "@/lib/training-control";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return Response.json(getTrainingSystemInfo());
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to detect training environment.";
    return Response.json({ error: message }, { status: 500 });
  }
}

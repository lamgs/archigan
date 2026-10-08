import { NextResponse } from "next/server";
import { generateRequestSchema } from "@/lib/contracts";
import { deriveMassing } from "@/lib/massing";
import { createMeshyPreview } from "@/lib/providers/meshy";

export async function POST(request: Request) {
  const parsed = generateRequestSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ error: "Invalid generation request.", issues: parsed.error.issues }, { status: 400 });
  const { prompt, refinement, provider } = parsed.data;
  if (provider === "procedural") return NextResponse.json({ kind: "massing", provider, massing: deriveMassing(prompt, refinement) });
  try {
    const taskId = await createMeshyPreview(prompt, refinement);
    return NextResponse.json({ kind: "task", provider, taskId, status: "PENDING", verified: false }, { status: 202 });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Hosted generation failed." }, { status: 503 });
  }
}

import { NextResponse } from "next/server";
import { generateRequestSchema } from "@/lib/contracts";
import { deriveMassing } from "@/lib/massing";
import { authorize, clientIp, guardResponse, providerErrorResponse, spendLimiter } from "@/lib/providers/hosted-http";
import { getProvider } from "@/lib/providers/registry";

export async function POST(request: Request) {
  const parsed = generateRequestSchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ error: "Invalid generation request.", issues: parsed.error.issues }, { status: 400 });
  const { prompt, refinement, provider, confirmSpend } = parsed.data;
  if (provider === "procedural") return NextResponse.json({ kind: "massing", provider, massing: deriveMassing(prompt, refinement) });

  // Paid path: configuration → access code → explicit confirmation → rate/daily limits → provider call.
  const adapter = getProvider(provider);
  const auth = authorize(request.headers, process.env, provider);
  if (!auth.ok) return guardResponse(auth);
  if (confirmSpend !== true) return NextResponse.json({ error: `Hosted generation spends ${adapter.label} credits and must be confirmed.`, code: "confirmation-required" }, { status: 400 });
  const ip = clientIp(request.headers);
  const limit = spendLimiter().check(ip, Date.now());
  if (!limit.ok) return guardResponse(limit);
  try {
    const taskId = await adapter.create(prompt, refinement);
    spendLimiter().record(ip, Date.now());
    return NextResponse.json({ kind: "task", provider, taskId, status: "queued", verified: false }, { status: 202 });
  } catch (error) {
    return providerErrorResponse(error);
  }
}

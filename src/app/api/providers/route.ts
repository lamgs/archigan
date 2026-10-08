import { NextResponse } from "next/server";
import { meshyStatus } from "@/lib/providers/meshy";

export const dynamic = "force-dynamic";

export function GET() {
  const { configured, enabled, hasKey, accessCodeRequired, verified } = meshyStatus();
  return NextResponse.json({ procedural: { configured: true, verified: true }, meshy: { configured, enabled, hasKey, accessCodeRequired, verified } });
}

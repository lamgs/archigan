import { NextResponse } from "next/server";
import { meshyStatus } from "@/lib/providers/meshy";

export const dynamic = "force-dynamic";

export function GET() {
  return NextResponse.json({ procedural: { configured: true, verified: true }, meshy: meshyStatus() });
}


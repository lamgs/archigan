import { NextResponse } from "next/server";
import { providerCatalog } from "@/lib/providers/registry";

export const dynamic = "force-dynamic";

/** Secret-free provider catalog: configured/enabled flags, approximate cost labels and cancel support. Every hosted provider is `verified: false`. */
export function GET() {
  return NextResponse.json({ procedural: { configured: true, verified: true }, ...providerCatalog(process.env) });
}

import { NextResponse } from "next/server";
import { providerErrorResponse, taskRequest } from "@/lib/providers/hosted-http";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ taskId: string }> };

/** Normalized task status; the browser polls this. The model URL is never returned — use `/model` to ingest the GLB. */
export async function GET(request: Request, { params }: Context) {
  const checked = await taskRequest(request, params);
  if ("response" in checked) return checked.response;
  try {
    const { glbUrl, ...task } = await checked.provider.status(checked.taskId);
    return NextResponse.json({ task: { ...task, hasModel: Boolean(glbUrl) } });
  } catch (error) {
    return providerErrorResponse(error);
  }
}

/** Cancels via the provider (where it supports cancel). A task the vendor refuses to cancel answers 409 (`running`). */
export async function DELETE(request: Request, { params }: Context) {
  const checked = await taskRequest(request, params);
  if ("response" in checked) return checked.response;
  try {
    await checked.provider.cancel(checked.taskId);
    return NextResponse.json({ ok: true, status: "cancelled" });
  } catch (error) {
    return providerErrorResponse(error);
  }
}

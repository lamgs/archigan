import { NextResponse } from "next/server";
import { providerErrorResponse, taskRequest } from "@/lib/providers/hosted-http";
import { deleteMeshyTask, getMeshyTask } from "@/lib/providers/meshy";

export const dynamic = "force-dynamic";

type Context = { params: Promise<{ taskId: string }> };

/** Normalized task status; the browser polls this. The model URL is never returned — use `/model` to ingest the GLB. */
export async function GET(request: Request, { params }: Context) {
  const checked = await taskRequest(request, params);
  if ("response" in checked) return checked.response;
  try {
    const { glbUrl, ...task } = await getMeshyTask(checked.taskId);
    return NextResponse.json({ task: { ...task, hasModel: Boolean(glbUrl) } });
  } catch (error) {
    return providerErrorResponse(error);
  }
}

/** Cancels by deleting. Meshy only allows this for queued or finished tasks; a running task answers 409 (`running`). */
export async function DELETE(request: Request, { params }: Context) {
  const checked = await taskRequest(request, params);
  if ("response" in checked) return checked.response;
  try {
    await deleteMeshyTask(checked.taskId);
    return NextResponse.json({ ok: true, status: "cancelled" });
  } catch (error) {
    return providerErrorResponse(error);
  }
}

import { providerErrorResponse, taskRequest } from "@/lib/providers/hosted-http";
import { downloadGlb, getMeshyTask, MeshyError } from "@/lib/providers/meshy";

export const dynamic = "force-dynamic";

/** Fetches a fresh task (so the signed URL is current) and streams the GLB to the browser for permanent local storage. */
export async function GET(request: Request, { params }: { params: Promise<{ taskId: string }> }) {
  const checked = await taskRequest(request, params);
  if ("response" in checked) return checked.response;
  try {
    const task = await getMeshyTask(checked.taskId);
    if (task.status !== "completed" || !task.glbUrl) throw new MeshyError("not-ready", "The model is not ready yet.", 409, true);
    const bytes = await downloadGlb(task.glbUrl);
    return new Response(bytes, { headers: { "Content-Type": "model/gltf-binary", "Content-Length": String(bytes.byteLength), "Cache-Control": "no-store" } });
  } catch (error) {
    return providerErrorResponse(error);
  }
}

import { composeArchitecturalPrompt } from "../massing";

const MESHY_URL = "https://api.meshy.ai/openapi/v2/text-to-3d";

export function meshyStatus() {
  const enabled = process.env.MESHY_ENABLED === "true";
  const hasKey = Boolean(process.env.MESHY_API_KEY);
  return { configured: enabled && hasKey, enabled, hasKey, verified: false as const };
}

export async function createMeshyPreview(prompt: string, refinement: string) {
  const status = meshyStatus();
  if (!status.configured) throw new Error("Meshy is not configured on the server.");
  const response = await fetch(MESHY_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.MESHY_API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ mode: "preview", prompt: composeArchitecturalPrompt(prompt, refinement), model_type: "standard", target_formats: ["glb"] }),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Meshy request failed with status ${response.status}.`);
  const data = (await response.json()) as { result?: string };
  if (!data.result) throw new Error("Meshy response did not include a task ID.");
  return data.result;
}


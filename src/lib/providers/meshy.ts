import { z } from "zod";
import { composeArchitecturalPrompt } from "../massing";
import { clampProgress, downloadGlbFor, isAllowedHost, mapHttpError as mapVendorHttpError, timedFetch } from "./http";
import { MAX_GLB_BYTES, ProviderError, type Fetch, type HostedProvider, type JobStatus, type NormalizedTask } from "./types";

export { DOWNLOAD_TIMEOUT_MS, MAX_GLB_BYTES, REQUEST_TIMEOUT_MS } from "./types";
export type { JobStatus, NormalizedTask, TaskError } from "./types";

/**
 * Meshy Text-to-3D (v2) adapter — server-side only.
 *
 * Contract source: Meshy's public API documentation (create: POST /openapi/v2/text-to-3d, retrieve:
 * GET /openapi/v2/text-to-3d/{id}, stream: GET .../{id}/stream, delete: DELETE .../{id}; task statuses PENDING,
 * IN_PROGRESS, SUCCEEDED, FAILED, CANCELED; DELETE on an IN_PROGRESS task returns 409; results are retained for a
 * limited time). This code has NOT been exercised against a live account, so response parsing is deliberately lenient
 * and `meshyStatus().verified` is always false until a real-key smoke test is recorded in docs/DEPLOYMENT.md.
 */
export const MESHY_URL = "https://api.meshy.ai/openapi/v2/text-to-3d";
export const MeshyError = ProviderError;
export type MeshyError = ProviderError;
const ASSET_HOSTS = ["meshy.ai"] as const;

export function meshyStatus(env: Record<string, string | undefined> = process.env) {
  const enabled = env.MESHY_ENABLED === "true";
  const hasKey = Boolean(env.MESHY_API_KEY);
  const accessCodeRequired = Boolean(env.SIFT_ACCESS_CODE || env.MESHY_ACCESS_CODE);
  // Fails closed: a key alone is not enough; the deployment must also define who may spend it.
  return { configured: enabled && hasKey && accessCodeRequired, enabled, hasKey, accessCodeRequired, verified: false as const };
}

const rawTaskSchema = z.object({
  id: z.string().optional(),
  status: z.string().optional(),
  progress: z.number().optional(),
  model_urls: z.object({ glb: z.string().optional() }).passthrough().optional(),
  task_error: z.object({ message: z.string().optional() }).passthrough().nullish(),
  expires_at: z.number().optional(),
  finished_at: z.number().optional(),
});

const STATUS_MAP: Record<string, JobStatus> = { PENDING: "queued", IN_PROGRESS: "running", SUCCEEDED: "completed", FAILED: "failed", CANCELED: "cancelled" };

/** Maps a raw Meshy task object to the provider-neutral shape. Unknown statuses are treated as an error, never as success. */
export function normalizeTask(raw: unknown, fallbackId = ""): NormalizedTask {
  const parsed = rawTaskSchema.safeParse(raw);
  if (!parsed.success) return { providerTaskId: fallbackId, status: "failed", error: { code: "bad-response", message: "Meshy returned a response this app does not understand.", retryable: false } };
  const task = parsed.data;
  const id = task.id ?? fallbackId;
  const status = STATUS_MAP[String(task.status ?? "").toUpperCase()];
  if (!status) return { providerTaskId: id, status: "failed", error: { code: "unknown-status", message: `Meshy reported an unrecognized status “${task.status ?? "none"}”.`, retryable: false } };
  const progress = clampProgress(task.progress);
  const glbUrl = task.model_urls?.glb || undefined;
  const expiresAt = task.expires_at ? new Date(task.expires_at).toISOString() : undefined;
  if (status === "failed") return { providerTaskId: id, status, ...(progress !== undefined && { progress }), error: { code: "provider-failed", message: task.task_error?.message || "Meshy could not generate this model.", retryable: true } };
  if (status === "completed" && !glbUrl) return { providerTaskId: id, status: "failed", error: { code: "no-glb", message: "Meshy finished but did not provide a GLB file.", retryable: false } };
  return { providerTaskId: id, status, ...(progress !== undefined && { progress }), ...(glbUrl && { glbUrl }), ...(expiresAt && { expiresAt }) };
}

export const mapHttpError = (status: number, retryAfter?: string | null) => mapVendorHttpError(status, retryAfter, "Meshy");

async function call(path: string, init: RequestInit, fetchImpl: Fetch): Promise<Response> {
  const key = process.env.MESHY_API_KEY;
  if (!key) throw new MeshyError("not-configured", "Meshy is not configured on the server.", 503, false);
  const response = await timedFetch("Meshy", `${MESHY_URL}${path}`, { ...init, headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...init.headers } }, fetchImpl);
  if (!response.ok) throw mapHttpError(response.status, response.headers.get("retry-after"));
  return response;
}

export async function createMeshyPreview(prompt: string, refinement: string, fetchImpl: Fetch = fetch): Promise<string> {
  const response = await call("", { method: "POST", body: JSON.stringify({ mode: "preview", prompt: composeArchitecturalPrompt(prompt, refinement), model_type: "standard", target_formats: ["glb"] }) }, fetchImpl);
  const data = (await response.json().catch(() => null)) as { result?: unknown } | null;
  if (typeof data?.result !== "string" || !data.result) throw new MeshyError("bad-response", "Meshy did not return a task ID.", 502, false);
  return data.result;
}

export async function getMeshyTask(taskId: string, fetchImpl: Fetch = fetch): Promise<NormalizedTask> {
  const response = await call(`/${encodeURIComponent(taskId)}`, { method: "GET" }, fetchImpl);
  return normalizeTask(await response.json().catch(() => null), taskId);
}

/** Meshy only deletes (and refunds) tasks that are PENDING or finished; a running task answers 409 → MeshyError("running"). */
export async function deleteMeshyTask(taskId: string, fetchImpl: Fetch = fetch): Promise<void> {
  await call(`/${encodeURIComponent(taskId)}`, { method: "DELETE" }, fetchImpl);
}

export const isAllowedAssetUrl = (value: string) => isAllowedHost(value, ASSET_HOSTS);

/** Downloads a GLB from a (signed) Meshy URL with a size cap and a magic-number check. */
export const downloadGlb = (url: string, fetchImpl: Fetch = fetch) => downloadGlbFor(meshyProvider, url, fetchImpl);

export const meshyProvider: HostedProvider = {
  id: "meshy",
  label: "Meshy",
  costLabel: "≈ 20 credits (≈ $0.40+) — estimate, plan-dependent",
  supportsCancel: true,
  assetHosts: ASSET_HOSTS,
  maxGlbBytes: MAX_GLB_BYTES,
  config: meshyStatus,
  create: (prompt, refinement, fetchImpl) => createMeshyPreview(prompt, refinement, fetchImpl),
  status: (taskId, fetchImpl) => getMeshyTask(taskId, fetchImpl),
  cancel: (taskId, fetchImpl) => deleteMeshyTask(taskId, fetchImpl),
};

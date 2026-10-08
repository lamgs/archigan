import { z } from "zod";
import { composeArchitecturalPrompt } from "../massing";

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
export const REQUEST_TIMEOUT_MS = 15_000;
export const DOWNLOAD_TIMEOUT_MS = 60_000;
export const MAX_GLB_BYTES = 100 * 1024 * 1024;

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled" | "timed-out" | "rate-limited";
export type TaskError = { code: string; message: string; retryable: boolean };
export type NormalizedTask = { providerTaskId: string; status: JobStatus; progress?: number; glbUrl?: string; expiresAt?: string; error?: TaskError };

export class MeshyError extends Error {
  constructor(readonly code: string, message: string, readonly httpStatus: number, readonly retryable: boolean, readonly retryAfterSeconds?: number) {
    super(message);
  }
}

export function meshyStatus() {
  const enabled = process.env.MESHY_ENABLED === "true";
  const hasKey = Boolean(process.env.MESHY_API_KEY);
  const accessCodeRequired = Boolean(process.env.MESHY_ACCESS_CODE);
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
  const progress = task.progress === undefined ? undefined : Math.max(0, Math.min(100, Math.round(task.progress)));
  const glbUrl = task.model_urls?.glb || undefined;
  const expiresAt = task.expires_at ? new Date(task.expires_at).toISOString() : undefined;
  if (status === "failed") return { providerTaskId: id, status, ...(progress !== undefined && { progress }), error: { code: "provider-failed", message: task.task_error?.message || "Meshy could not generate this model.", retryable: true } };
  if (status === "completed" && !glbUrl) return { providerTaskId: id, status: "failed", error: { code: "no-glb", message: "Meshy finished but did not provide a GLB file.", retryable: false } };
  return { providerTaskId: id, status, ...(progress !== undefined && { progress }), ...(glbUrl && { glbUrl }), ...(expiresAt && { expiresAt }) };
}

export function mapHttpError(status: number, retryAfter?: string | null): MeshyError {
  const seconds = retryAfter && Number.isFinite(Number(retryAfter)) ? Math.max(1, Math.round(Number(retryAfter))) : undefined;
  if (status === 401 || status === 403) return new MeshyError("auth", "Meshy rejected the server's API key.", status, false);
  if (status === 402) return new MeshyError("insufficient-credits", "The Meshy account has no credits left for this request.", status, false);
  if (status === 404) return new MeshyError("not-found", "Meshy no longer has this task (results are only retained for a limited time).", status, false);
  if (status === 409) return new MeshyError("running", "Meshy cannot cancel a task that is already running.", status, false);
  if (status === 429) return new MeshyError("rate-limited", "Meshy is rate limiting requests. Try again shortly.", status, true, seconds ?? 10);
  if (status === 400 || status === 422) return new MeshyError("rejected", "Meshy rejected this request.", status, false);
  if (status >= 500) return new MeshyError("provider-unavailable", "Meshy is temporarily unavailable.", status, true, seconds);
  return new MeshyError("unexpected", `Meshy responded with status ${status}.`, status, false);
}

type Fetch = typeof fetch;

async function call(path: string, init: RequestInit, fetchImpl: Fetch): Promise<Response> {
  const key = process.env.MESHY_API_KEY;
  if (!key) throw new MeshyError("not-configured", "Meshy is not configured on the server.", 503, false);
  let response: Response;
  try {
    response = await fetchImpl(`${MESHY_URL}${path}`, { ...init, headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...init.headers }, cache: "no-store", signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
  } catch (error) {
    const timeout = error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError");
    throw new MeshyError(timeout ? "timeout" : "network", timeout ? "Meshy did not respond in time." : "Could not reach Meshy.", 504, true);
  }
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

/** Only HTTPS URLs on Meshy-owned hosts may be fetched on the user's behalf (prevents the server being used as an open proxy). */
export function isAllowedAssetUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === "https:" && (url.hostname === "meshy.ai" || url.hostname.endsWith(".meshy.ai"));
  } catch {
    return false;
  }
}

/** Downloads a GLB from a (signed) Meshy URL with a size cap and a magic-number check. */
export async function downloadGlb(url: string, fetchImpl: Fetch = fetch): Promise<ArrayBuffer> {
  if (!isAllowedAssetUrl(url)) throw new MeshyError("bad-asset-url", "Meshy returned a model URL this app will not fetch.", 502, false);
  let response: Response;
  try {
    response = await fetchImpl(url, { cache: "no-store", signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
  } catch {
    throw new MeshyError("network", "Could not download the model from Meshy.", 504, true);
  }
  if (response.status === 403 || response.status === 410) throw new MeshyError("asset-expired", "The signed download link has expired. Re-check the task for a fresh link.", response.status, true);
  if (!response.ok) throw mapHttpError(response.status);
  const declared = Number(response.headers.get("content-length") ?? 0);
  if (declared > MAX_GLB_BYTES) throw new MeshyError("too-large", "The model file is too large to import.", 502, false);
  const bytes = await response.arrayBuffer();
  if (bytes.byteLength > MAX_GLB_BYTES) throw new MeshyError("too-large", "The model file is too large to import.", 502, false);
  if (bytes.byteLength < 12 || new TextDecoder().decode(new Uint8Array(bytes, 0, 4)) !== "glTF") throw new MeshyError("not-glb", "The downloaded file is not a valid GLB.", 502, false);
  return bytes;
}

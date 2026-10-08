import { z } from "zod";
import { composeArchitecturalPrompt } from "../massing";
import { clampProgress, hostedConfig, mapHttpError, timedFetch } from "./http";
import { MAX_GLB_BYTES, ProviderError, type Fetch, type HostedProvider, type JobStatus, type NormalizedTask } from "./types";

/**
 * Tripo text-to-model adapter — server-side only.
 *
 * EVERYTHING BELOW IS UNVERIFIED: Tripo's official docs (platform.tripo3d.ai / docs.tripo3d.ai) could not be fetched
 * when this was written; the contract comes from search snippets and recollection of the v2 API. It has NOT been
 * exercised against a live account, so parsing is deliberately lenient and `config().verified` is always false.
 *
 * Assumed v2 contract (UNVERIFIED):
 *  - Base https://api.tripo3d.ai/v2/openapi, `Authorization: Bearer $TRIPO_API_KEY`.
 *  - Create: POST /task {type:"text_to_model", prompt} -> {code:0, data:{task_id}}.
 *  - Status: GET /task/{task_id} -> {code:0, data:{task_id,type,status,progress,output:{model,pbr_model,base_model,rendered_image}}}
 *    with status queued|running|success|failed|cancelled|unknown|banned|expired. Newer docs mention `output.model_url`
 *    (sample file model_pbr.glb); we accept the first of output.pbr_model / output.model / output.model_url (string or {url}).
 *  - AMBIGUITY: Tripo's newer "v3" docs describe dedicated per-capability endpoints (keys reportedly shared between v2/v3).
 *    We keep the v2 single-endpoint contract; change TRIPO_BASE_URL / TRIPO_TASK_PATH below if that proves wrong.
 *  - A non-zero envelope `code` is an error (2010 ~ insufficient credits, 2000 ~ rate limit; mapped leniently).
 *  - No cancel endpoint is known, so cancel is unsupported and never touches the network.
 *  - Pricing could not be confirmed; no numbers are invented.
 */
export const TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi";
export const TRIPO_TASK_PATH = "/task";
const ASSET_HOSTS = ["tripo3d.com", "tripo3d.ai"] as const;
const MAX_PROMPT_CHARS = 1024; // conservative; Tripo's real limit is unconfirmed
const TASK_ID = /^[A-Za-z0-9_-]{6,80}$/;

const urlish = z.union([z.string(), z.object({ url: z.string().optional() }).passthrough()]).nullish();
const envelopeSchema = z.object({
  code: z.number().optional(),
  message: z.string().optional(),
  data: z.object({
    task_id: z.string().optional(),
    status: z.string().optional(),
    progress: z.number().optional(),
    output: z.object({ pbr_model: urlish, model: urlish, model_url: urlish }).passthrough().nullish(),
  }).passthrough().nullish(),
}).passthrough();
type Envelope = z.infer<typeof envelopeSchema>;

const STATUS_MAP: Record<string, JobStatus> = { queued: "queued", running: "running", success: "completed", cancelled: "cancelled" };

export function truncatePrompt(text: string, max = MAX_PROMPT_CHARS): string {
  if (text.length <= max) return text;
  let cut = text.slice(0, max);
  const last = cut.charCodeAt(cut.length - 1);
  if (last >= 0xd800 && last <= 0xdbff) cut = cut.slice(0, -1); // never split a surrogate pair
  return cut.trimEnd();
}

const failure = (id: string, code: string, message: string, retryable = false): NormalizedTask => ({ providerTaskId: id, status: "failed", error: { code, message, retryable } });

function urlOf(value: unknown): string | undefined {
  const url = typeof value === "string" ? value : value && typeof value === "object" ? (value as { url?: unknown }).url : undefined;
  return typeof url === "string" && url ? url : undefined;
}

/** Best-effort expiry from a signed URL's `Expires=` epoch-seconds param (UNVERIFIED that Tripo uses it). */
function expiryOf(url: string): string | undefined {
  try {
    const raw = new URL(url).searchParams.get("Expires");
    const seconds = raw && /^\d{9,11}$/.test(raw) ? Number(raw) : undefined;
    return seconds ? new Date(seconds * 1000).toISOString() : undefined;
  } catch {
    return undefined;
  }
}

export function normalizeTripoTask(raw: unknown, fallbackId = ""): NormalizedTask {
  const parsed = envelopeSchema.safeParse(raw);
  if (!parsed.success || !parsed.data.data) return failure(fallbackId, "bad-response", "Tripo returned a response this app does not understand.");
  const data = parsed.data.data;
  const id = data.task_id ?? fallbackId;
  const raw_status = String(data.status ?? "").toLowerCase();
  const progress = clampProgress(data.progress);
  const withProgress = progress !== undefined ? { progress } : {};
  if (raw_status === "failed" || raw_status === "banned") return { providerTaskId: id, status: "failed", ...withProgress, error: { code: raw_status === "banned" ? "banned" : "provider-failed", message: raw_status === "banned" ? "Tripo declined to generate this model." : "Tripo could not generate this model.", retryable: raw_status === "failed" } };
  if (raw_status === "expired") return failure(id, "expired", "Tripo no longer retains this task's results.");
  if (raw_status === "unknown") return failure(id, "unknown-status", "Tripo reported an unrecognized status “unknown”.");
  const status = STATUS_MAP[raw_status];
  if (!status) return failure(id, "unknown-status", `Tripo reported an unrecognized status “${data.status ?? "none"}”.`);
  if (status === "completed") {
    const out = data.output;
    const glbUrl = urlOf(out?.pbr_model) ?? urlOf(out?.model) ?? urlOf(out?.model_url);
    if (!glbUrl) return failure(id, "no-glb", "Tripo finished but did not provide a GLB file.");
    const expiresAt = expiryOf(glbUrl);
    return { providerTaskId: id, status, progress: progress ?? 100, glbUrl, ...(expiresAt && { expiresAt }) };
  }
  return { providerTaskId: id, status, ...withProgress };
}

/** Maps a non-zero envelope code leniently (UNVERIFIED code numbers). */
function envelopeError(body: Envelope): ProviderError {
  const code = body.code;
  if (code === 2010) return new ProviderError("insufficient-credits", "The Tripo account has no credits left for this request.", 402, false);
  if (code === 2000) return new ProviderError("rate-limited", "Tripo is rate limiting requests. Try again shortly.", 429, true, 10);
  return new ProviderError("provider-error", `Tripo rejected the request (code ${String(code)}).`, 502, false);
}

async function call(path: string, init: RequestInit, fetchImpl: Fetch, env: Record<string, string | undefined>): Promise<Envelope> {
  const key = env.TRIPO_API_KEY;
  if (!key) throw new ProviderError("not-configured", "Tripo is not configured on the server.", 503, false);
  const response = await timedFetch("Tripo", `${TRIPO_BASE_URL}${path}`, { ...init, headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...init.headers } }, fetchImpl);
  if (!response.ok) throw mapHttpError(response.status, response.headers.get("retry-after"), "Tripo");
  const body = envelopeSchema.safeParse(await response.json().catch(() => null));
  if (!body.success) throw new ProviderError("bad-response", "Tripo returned a response this app does not understand.", 502, false);
  if (body.data.code !== undefined && body.data.code !== 0) throw envelopeError(body.data);
  return body.data;
}

export const tripoProvider: HostedProvider = {
  id: "tripo",
  label: "Tripo",
  costLabel: "cost not confirmed — check your Tripo plan",
  supportsCancel: false,
  assetHosts: ASSET_HOSTS,
  maxGlbBytes: MAX_GLB_BYTES,
  config: (env) => hostedConfig(env, "TRIPO_ENABLED", ["TRIPO_API_KEY"]),
  async create(prompt, refinement, fetchImpl = fetch) {
    const body = { type: "text_to_model", prompt: truncatePrompt(composeArchitecturalPrompt(prompt, refinement)) };
    const result = await call(TRIPO_TASK_PATH, { method: "POST", body: JSON.stringify(body) }, fetchImpl, process.env);
    const id = result.data?.task_id;
    if (typeof id !== "string" || !TASK_ID.test(id)) throw new ProviderError("bad-response", "Tripo did not return a usable task ID.", 502, false);
    return id;
  },
  async status(taskId, fetchImpl = fetch) {
    if (!TASK_ID.test(taskId)) throw new ProviderError("bad-response", "Invalid Tripo task ID.", 502, false);
    const result = await call(`${TRIPO_TASK_PATH}/${encodeURIComponent(taskId)}`, { method: "GET" }, fetchImpl, process.env);
    return normalizeTripoTask(result, taskId);
  },
  async cancel() {
    throw new ProviderError("cancel-unsupported", "Tripo does not document a way to cancel a task.", 501, false);
  },
};

import { z } from "zod";
import { composeArchitecturalPrompt, normalizeBrief } from "../massing";
import { hostedConfig, mapHttpError, timedFetch } from "./http";
import { MAX_GLB_BYTES, ProviderError, type Fetch, type HostedProvider, type JobStatus, type NormalizedTask } from "./types";

/**
 * Hunyuan3D v3.1 via the fal.ai queue API — server-side only.
 *
 * EVERYTHING BELOW IS UNVERIFIED. The official fal.ai docs were not reachable when this was written; the contract is
 * reconstructed from search snippets and has never been exercised against a live account:
 *  - UNVERIFIED endpoints: `fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d` (prompt <= 200 chars; ≈ $0.225) and
 *    `fal-ai/hunyuan-3d/v3.1/pro/text-to-3d` (prompt <= 1024 chars; ≈ $0.375). Prices are estimates.
 *  - UNVERIFIED submit: POST https://queue.fal.run/{endpoint}, `Authorization: Key $FAL_KEY`, JSON {prompt, ...}
 *    -> {request_id, status_url, response_url, cancel_url}.
 *  - UNVERIFIED status/result/cancel: fal addresses these by APP id (first two path segments, `fal-ai/hunyuan-3d`),
 *    not the full endpoint path: GET .../requests/{id}/status, GET .../requests/{id}, PUT .../requests/{id}/cancel.
 *    Statuses IN_QUEUE | IN_PROGRESS | COMPLETED. Cancel answers 400 ALREADY_COMPLETED once finished.
 *  - UNVERIFIED output: `model_glb {url, content_type, file_name}` (also tolerating `model_urls.glb.url`).
 *  - UNVERIFIED: a failed run surfaces as an `error`/`detail` field in the COMPLETED result or a 422 on the result GET.
 * fal reports no progress percentage, so `progress` is left undefined. `verified` is always false (see hostedConfig).
 */
const QUEUE_BASE = "https://queue.fal.run";
const VENDOR = "Hunyuan3D (fal.ai)";
const ASSET_HOSTS = ["fal.media"] as const; // suffix match also covers v3.fal.media etc.
const TASK_ID = /^[A-Za-z0-9_-]{6,80}$/;

type Tier = { id: "hunyuan3d-rapid" | "hunyuan3d-pro"; label: string; endpoint: string; promptLimit: number; costLabel: string; options: Record<string, unknown> };

const TIERS: Tier[] = [
  { id: "hunyuan3d-rapid", label: "Hunyuan3D Rapid", endpoint: "fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d", promptLimit: 200, costLabel: "≈ $0.225 per model (estimate)", options: { enable_pbr: false, enable_geometry: false } },
  { id: "hunyuan3d-pro", label: "Hunyuan3D Pro", endpoint: "fal-ai/hunyuan-3d/v3.1/pro/text-to-3d", promptLimit: 1024, costLabel: "≈ $0.375 per model (estimate)", options: { face_count: 200_000 } },
];

/** fal's queue status/result/cancel routes use the app id: the first two path segments of the endpoint. */
export const falAppId = (endpoint: string) => endpoint.split("/").slice(0, 2).join("/");

const utf8 = new TextEncoder();
/** Cuts to at most `limit` UTF-8 bytes without splitting a code point (conservative for either bytes or characters). */
export function truncateUtf8(text: string, limit: number): string {
  let out = "";
  let bytes = 0;
  for (const ch of text) {
    const n = utf8.encode(ch).length;
    if (bytes + n > limit) break;
    out += ch;
    bytes += n;
  }
  return out.trimEnd();
}

/** Full architectural prompt if it fits; otherwise a compact template; always hard-cut to the tier's limit. */
export function composeHunyuanPrompt(prompt: string, refinement: string, limit: number): string {
  const full = composeArchitecturalPrompt(prompt, refinement);
  if (utf8.encode(full).length <= limit) return full;
  const compact = `Architectural massing model of ${normalizeBrief(prompt, refinement)}. Standalone building, clean geometry.`;
  return truncateUtf8(utf8.encode(compact).length <= limit ? compact : compact.replace(/\. Standalone building, clean geometry\.$/, ""), limit);
}

const submitSchema = z.object({ request_id: z.string().optional() }).passthrough();
const statusSchema = z.object({ status: z.string().optional(), error: z.unknown().optional() }).passthrough();
const fileSchema = z.object({ url: z.string().optional(), content_type: z.string().nullish(), file_name: z.string().nullish() }).passthrough();
const resultSchema = z.object({
  model_glb: fileSchema.nullish(),
  model_urls: z.object({ glb: fileSchema.nullish() }).passthrough().nullish(),
  error: z.unknown().optional(),
  detail: z.unknown().optional(),
}).passthrough();

const STATUS_MAP: Record<string, JobStatus> = { IN_QUEUE: "queued", IN_PROGRESS: "running" };
const fail = (id: string, code: string, message: string, retryable = false): NormalizedTask => ({ providerTaskId: id, status: "failed", error: { code, message, retryable } });

function getKey(): string {
  const key = process.env.FAL_KEY;
  if (!key) throw new ProviderError("not-configured", "Hunyuan3D is not configured on the server.", 503, false);
  return key;
}

async function call(url: string, init: RequestInit, fetchImpl: Fetch): Promise<Response> {
  const key = getKey();
  const response = await timedFetch(VENDOR, url, { ...init, headers: { Authorization: `Key ${key}`, "Content-Type": "application/json", ...init.headers } }, fetchImpl);
  if (response.ok) return response;
  if (response.status === 403) {
    // fal reports an exhausted balance as 403; the body text is only inspected, never echoed.
    const text = await response.text().catch(() => "");
    if (/balance|credit|billing|insufficient/i.test(text)) throw mapHttpError(402, null, VENDOR);
  }
  throw mapHttpError(response.status, response.headers.get("retry-after"), VENDOR);
}

const safeId = (taskId: string) => {
  if (!TASK_ID.test(taskId)) throw new ProviderError("bad-task-id", "That task ID is not valid.", 400, false);
  return taskId;
};

function extractGlb(raw: unknown, id: string): NormalizedTask {
  const parsed = resultSchema.safeParse(raw);
  if (!parsed.success) return fail(id, "bad-response", `${VENDOR} returned a response this app does not understand.`);
  const result = parsed.data;
  if (result.error || result.detail) return fail(id, "provider-failed", `${VENDOR} could not generate this model.`, true);
  const file = result.model_glb ?? result.model_urls?.glb;
  if (!file?.url) return fail(id, "no-glb", `${VENDOR} finished but did not provide a GLB file.`);
  const type = (file.content_type ?? "").toLowerCase();
  const name = (file.file_name ?? "").toLowerCase();
  let path = "";
  try { path = new URL(file.url).pathname.toLowerCase(); } catch { /* validated later by the host allowlist */ }
  const looksGlb = !name.endsWith(".obj") && (type.includes("gltf-binary") || type.includes("model/glb") || name.endsWith(".glb") || (!name && path.endsWith(".glb")));
  if (!looksGlb) return fail(id, "not-glb", `${VENDOR} returned a non-GLB model format; this app only imports GLB.`);
  return { providerTaskId: id, status: "completed", glbUrl: file.url };
}

function makeProvider(tier: Tier): HostedProvider {
  const app = `${QUEUE_BASE}/${falAppId(tier.endpoint)}/requests`;
  return {
    id: tier.id,
    label: tier.label,
    costLabel: tier.costLabel,
    supportsCancel: true,
    assetHosts: ASSET_HOSTS,
    maxGlbBytes: MAX_GLB_BYTES,
    config: (env) => hostedConfig(env, "HUNYUAN_ENABLED", ["FAL_KEY"]),
    async create(prompt, refinement, fetchImpl = fetch) {
      const body = { prompt: composeHunyuanPrompt(prompt, refinement, tier.promptLimit), ...tier.options };
      const response = await call(`${QUEUE_BASE}/${tier.endpoint}`, { method: "POST", body: JSON.stringify(body) }, fetchImpl);
      const data = submitSchema.safeParse(await response.json().catch(() => null));
      const id = data.success ? data.data.request_id : undefined;
      if (typeof id !== "string" || !TASK_ID.test(id)) throw new ProviderError("bad-response", `${VENDOR} did not return a usable task ID.`, 502, false);
      return id;
    },
    async status(taskId, fetchImpl = fetch) {
      const id = safeId(taskId);
      const response = await call(`${app}/${id}/status`, { method: "GET" }, fetchImpl);
      const parsed = statusSchema.safeParse(await response.json().catch(() => null));
      if (!parsed.success) return fail(id, "bad-response", `${VENDOR} returned a response this app does not understand.`);
      const raw = String(parsed.data.status ?? "").toUpperCase();
      if (raw === "COMPLETED") {
        if (parsed.data.error) return fail(id, "provider-failed", `${VENDOR} could not generate this model.`, true);
        let result: Response;
        try {
          result = await call(`${app}/${id}`, { method: "GET" }, fetchImpl);
        } catch (error) {
          // A failed app run is reported by fal as a 422 on the result route (UNVERIFIED).
          if (error instanceof ProviderError && error.httpStatus === 422) return fail(id, "provider-failed", `${VENDOR} could not generate this model.`, true);
          throw error;
        }
        return extractGlb(await result.json().catch(() => null), id);
      }
      const status = STATUS_MAP[raw];
      if (!status) return fail(id, "unknown-status", `${VENDOR} reported an unrecognized status “${parsed.data.status ?? "none"}”.`);
      return { providerTaskId: id, status };
    },
    async cancel(taskId, fetchImpl = fetch) {
      const id = safeId(taskId);
      try {
        await call(`${app}/${id}/cancel`, { method: "PUT" }, fetchImpl);
      } catch (error) {
        if (error instanceof ProviderError && error.httpStatus === 400) throw new ProviderError("already-completed", `${VENDOR} already finished this task, so it cannot be cancelled.`, 409, false);
        throw error;
      }
    },
  };
}

export const hunyuanRapidProvider: HostedProvider = makeProvider(TIERS[0]);
export const hunyuanProProvider: HostedProvider = makeProvider(TIERS[1]);

import { createHash, createHmac } from "node:crypto";
import { z } from "zod";
import { hostedConfig, mapHttpError, timedFetch } from "./http";
import { composeHunyuanPrompt } from "./hunyuan";
import { MAX_GLB_BYTES, ProviderError, type Fetch, type HostedProvider, type JobStatus, type NormalizedTask } from "./types";

/**
 * Tencent Cloud AI3D (Hunyuan 3D) text-to-3D, called directly with TC3-HMAC-SHA256 (no SDK) — server-side only.
 *
 * THE WHOLE CONTRACT IS UNVERIFIED: it has never been exercised against a live account. Sources, by reliability:
 *  - Read from the official Go SDK source (github.com/TencentCloud/tencentcloud-sdk-go, ai3d/v20250513/models.go, raw GitHub):
 *    service `ai3d`, version `2025-05-13`; actions SubmitHunyuanTo3DRapidJob / QueryHunyuanTo3DRapidJob and
 *    SubmitHunyuanTo3DProJob / QueryHunyuanTo3DProJob; request `Prompt` (text-to-3D; exclusive with image inputs;
 *    Rapid <= 200, Pro <= 1024 "UTF-8 characters", documented as a Chinese description); Rapid `ResultFormat`
 *    (OBJ default | GLB | STL | USDZ | FBX | MP4), `EnablePBR`; Pro `EnablePBR`, `FaceCount` (3000-1500000),
 *    `GenerateType` (Normal default), `Model` (3.0 | 3.1); Pro `ResultFormat` only offers STL/USDZ/FBX and by default
 *    returns OBJ + GLB (so it is not sent); response `JobId` (valid 24h); query `Status` WAIT|RUN|FAIL|DONE, `ErrorCode`,
 *    `ErrorMessage`, `ResultFile3Ds[{Type, Url, PreviewImageUrl}]` (URL valid 24h). Default concurrency is 1 job.
 *  - From search snippets only (unconfirmed): host ai3d.intl.tencentcloudapi.com (international site), response envelope
 *    {"Response":{..., "RequestId", "Error":{"Code","Message"}}}, region ap-guangzhou used by third-party clients,
 *    Type values GLB/OBJ, error-code families (AuthFailure.*, RequestLimitExceeded*, ...). One third-party note claims Rapid is
 *    image-only; the SDK doc and another client show Rapid text prompts, so we send them and surface any rejection.
 *  - Not found anywhere: job-id character set (we require /^[A-Za-z0-9_-]{6,80}$/), real result hosts (COS domains below are
 *    inferred), cancel API (none documented => unsupported), official signature vector (secret keys are masked in docs).
 *  - Costs are estimates from explainer articles (credits ≈ $0.0135-0.015 each), not a quote.
 * `verified` is always false (see hostedConfig).
 */
const HOST = "ai3d.intl.tencentcloudapi.com";
const SERVICE = "ai3d";
const VERSION = "2025-05-13";
const REGION = "ap-guangzhou";
const VENDOR = "Tencent Cloud";
const TASK_ID = /^[A-Za-z0-9_-]{6,80}$/;
// Inferred COS result hosts (suffix match): myqcloud.com (documented COS domain), tencentcos.cn/.com (COS's newer default domains).
const ASSET_HOSTS = ["myqcloud.com", "tencentcos.cn", "tencentcos.com"] as const;

const sha256Hex = (data: string) => createHash("sha256").update(data, "utf8").digest("hex");
const hmac = (key: string | Buffer, data: string) => createHmac("sha256", key).update(data, "utf8").digest();

export type Tc3Input = { secretId: string; secretKey: string; service: string; host: string; action: string; version: string; region?: string; payload: string; timestamp: number };

/** Pure TC3-HMAC-SHA256 signer. The timestamp (unix seconds) is injected so the output is deterministic. */
export function signTc3(input: Tc3Input) {
  const date = new Date(input.timestamp * 1000).toISOString().slice(0, 10);
  const contentType = "application/json; charset=utf-8";
  const signedHeaders = "content-type;host;x-tc-action";
  const canonicalRequest = ["POST", "/", "", `content-type:${contentType}\nhost:${input.host}\nx-tc-action:${input.action.toLowerCase()}\n`, signedHeaders, sha256Hex(input.payload)].join("\n");
  const scope = `${date}/${input.service}/tc3_request`;
  const stringToSign = ["TC3-HMAC-SHA256", String(input.timestamp), scope, sha256Hex(canonicalRequest)].join("\n");
  const signature = hmac(hmac(hmac(hmac(`TC3${input.secretKey}`, date), input.service), "tc3_request"), stringToSign).toString("hex");
  const authorization = `TC3-HMAC-SHA256 Credential=${input.secretId}/${scope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;
  const headers: Record<string, string> = { Authorization: authorization, "Content-Type": contentType, Host: input.host, "X-TC-Action": input.action, "X-TC-Version": input.version, "X-TC-Timestamp": String(input.timestamp) };
  if (input.region) headers["X-TC-Region"] = input.region;
  return { authorization, signature, canonicalRequest, stringToSign, headers };
}

type Tier = { id: "tencent-rapid" | "tencent-pro"; label: string; costLabel: string; promptLimit: number; submit: string; query: string; options: Record<string, unknown> };
const TIERS: Tier[] = [
  { id: "tencent-rapid", label: "HY 3D Rapid (Tencent)", costLabel: "≈ 15 credits ≈ $0.20–0.23 per model (estimate)", promptLimit: 200, submit: "SubmitHunyuanTo3DRapidJob", query: "QueryHunyuanTo3DRapidJob", options: { ResultFormat: "GLB", EnablePBR: false } },
  { id: "tencent-pro", label: "HY 3D Pro (Tencent)", costLabel: "≈ 25 credits ≈ $0.34–0.38 per model (estimate)", promptLimit: 1024, submit: "SubmitHunyuanTo3DProJob", query: "QueryHunyuanTo3DProJob", options: { GenerateType: "Normal", EnablePBR: false, FaceCount: 200_000 } },
];

const fileSchema = z.object({ Type: z.string().nullish(), Url: z.string().nullish() }).passthrough();
const envelopeSchema = z.object({ Response: z.object({
  Error: z.object({ Code: z.string().optional() }).passthrough().nullish(),
  JobId: z.string().nullish(),
  Status: z.string().nullish(),
  ErrorCode: z.string().nullish(),
  ResultFile3Ds: z.array(fileSchema).nullish(),
}).passthrough() }).passthrough();
type Body = z.infer<typeof envelopeSchema>["Response"];

/** Lenient Tencent error-code mapping. The code (never the message/body) selects a generic message. */
export function mapTencentError(code: string): ProviderError {
  if (/^AuthFailure/i.test(code)) return new ProviderError("auth", `${VENDOR} rejected the server's credentials.`, 401, false);
  if (/^(RequestLimitExceeded|LimitExceeded)/i.test(code)) return new ProviderError("rate-limited", `${VENDOR} is rate limiting requests. Try again shortly.`, 429, true, 10);
  if (/^ResourceInsufficient|balance|credit|arrear/i.test(code)) return new ProviderError("insufficient-credits", `The ${VENDOR} account has no credits left for this request.`, 402, false);
  if (/^InvalidParameter/i.test(code)) return new ProviderError("rejected", `${VENDOR} rejected this request.`, 400, false);
  if (/^(InternalError|ServiceUnavailable)/i.test(code)) return new ProviderError("provider-unavailable", `${VENDOR} is temporarily unavailable.`, 503, true);
  return new ProviderError("rejected", `${VENDOR} could not process this request.`, 502, false);
}

function credentials() {
  const secretId = process.env.TENCENT_SECRET_ID;
  const secretKey = process.env.TENCENT_SECRET_KEY;
  if (!secretId || !secretKey) throw new ProviderError("not-configured", "Tencent Cloud is not configured on the server.", 503, false);
  return { secretId, secretKey };
}

async function call(action: string, params: Record<string, unknown>, fetchImpl: Fetch): Promise<Body> {
  const payload = JSON.stringify(params);
  const { headers } = signTc3({ ...credentials(), service: SERVICE, host: HOST, action, version: VERSION, region: REGION, payload, timestamp: Math.floor(Date.now() / 1000) });
  const response = await timedFetch(VENDOR, `https://${HOST}/`, { method: "POST", headers, body: payload }, fetchImpl);
  const parsed = envelopeSchema.safeParse(await response.json().catch(() => null));
  const code = parsed.success ? parsed.data.Response.Error?.Code : undefined;
  if (code) throw mapTencentError(code);
  if (!response.ok) throw mapHttpError(response.status, response.headers.get("retry-after"), VENDOR);
  if (!parsed.success) throw new ProviderError("bad-response", `${VENDOR} returned a response this app does not understand.`, 502, false);
  return parsed.data.Response;
}

const safeId = (taskId: string) => {
  if (!TASK_ID.test(taskId)) throw new ProviderError("bad-task-id", "That task ID is not valid.", 400, false);
  return taskId;
};
const fail = (id: string, code: string, message: string, retryable = false): NormalizedTask => ({ providerTaskId: id, status: "failed", error: { code, message, retryable } });
const STATUS_MAP: Record<string, JobStatus> = { WAIT: "queued", RUN: "running" };

function extractGlb(body: Body, id: string): NormalizedTask {
  const files = (body.ResultFile3Ds ?? []).filter((f) => f.Url);
  const isGlb = (f: { Type?: string | null; Url?: string | null }) => {
    if ((f.Type ?? "").toUpperCase() === "GLB") return true;
    if (f.Type) return false;
    try { return new URL(f.Url ?? "").pathname.toLowerCase().endsWith(".glb"); } catch { return false; }
  };
  const glb = files.find(isGlb);
  if (glb?.Url) return { providerTaskId: id, status: "completed", glbUrl: glb.Url, expiresAt: new Date(Date.now() + 24 * 3600_000).toISOString() };
  if (files.length) return fail(id, "not-glb", `${VENDOR} returned a non-GLB model format; this app only imports GLB.`);
  return fail(id, "no-glb", `${VENDOR} finished but did not provide a GLB file.`);
}

function makeProvider(tier: Tier): HostedProvider {
  return {
    id: tier.id,
    label: tier.label,
    costLabel: tier.costLabel,
    supportsCancel: false,
    assetHosts: ASSET_HOSTS,
    maxGlbBytes: MAX_GLB_BYTES,
    config: (env) => hostedConfig(env, "TENCENT_HY3D_ENABLED", ["TENCENT_SECRET_ID", "TENCENT_SECRET_KEY"]),
    async create(prompt, refinement, fetchImpl = fetch) {
      const body = await call(tier.submit, { Prompt: composeHunyuanPrompt(prompt, refinement, tier.promptLimit), ...tier.options }, fetchImpl);
      const id = body.JobId;
      if (typeof id !== "string" || !TASK_ID.test(id)) throw new ProviderError("bad-response", `${VENDOR} did not return a usable task ID.`, 502, false);
      return id;
    },
    async status(taskId, fetchImpl = fetch) {
      const id = safeId(taskId);
      const body = await call(tier.query, { JobId: id }, fetchImpl);
      const raw = String(body.Status ?? "").toUpperCase();
      if (raw === "DONE") return extractGlb(body, id);
      if (raw === "FAIL") return fail(id, "provider-failed", `${VENDOR} could not generate this model.`, true);
      const status = STATUS_MAP[raw];
      if (!status) return fail(id, "unknown-status", `${VENDOR} reported an unrecognized status.`);
      return { providerTaskId: id, status };
    },
    async cancel() {
      throw new ProviderError("cancel-unsupported", `${VENDOR} does not document a way to cancel a task.`, 501, false);
    },
  };
}

export const tencentRapidProvider: HostedProvider = makeProvider(TIERS[0]);
export const tencentProProvider: HostedProvider = makeProvider(TIERS[1]);

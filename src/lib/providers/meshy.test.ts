import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { authorize, codeMatches, SpendLimiter } from "./guard";
import { createMeshyPreview, deleteMeshyTask, downloadGlb, getMeshyTask, isAllowedAssetUrl, mapHttpError, MeshyError, meshyStatus, normalizeTask, MESHY_URL } from "./meshy";

// NOTE: these are documented-contract tests against a mocked fetch. They do not prove live Meshy behaviour.
const json = (body: unknown, init: ResponseInit = {}) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" }, ...init });
const glb = (extra = 20) => { const b = new Uint8Array(12 + extra); b.set([0x67, 0x6c, 0x54, 0x46]); return b.buffer; };

beforeEach(() => { vi.stubEnv("MESHY_ENABLED", "true"); vi.stubEnv("MESHY_API_KEY", "msy_secret_key"); vi.stubEnv("MESHY_ACCESS_CODE", "letmein"); });
afterEach(() => vi.unstubAllEnvs());

describe("configuration", () => {
  it("is configured only with enable flag, key, and access code", () => {
    expect(meshyStatus()).toMatchObject({ configured: true, verified: false });
    vi.stubEnv("MESHY_ACCESS_CODE", "");
    expect(meshyStatus()).toMatchObject({ configured: false, hasKey: true, accessCodeRequired: false });
    vi.stubEnv("MESHY_ACCESS_CODE", "x"); vi.stubEnv("MESHY_ENABLED", "false");
    expect(meshyStatus().configured).toBe(false);
  });
  it("never reports verified, whatever the environment", () => expect(meshyStatus().verified).toBe(false));
});

describe("createMeshyPreview", () => {
  it("POSTs the documented v2 preview request with a bearer key and returns the task id", async () => {
    const fetchMock = vi.fn(async (..._args: unknown[]) => json({ result: "018a-task-1" }));
    await expect(createMeshyPreview("A tower", "with terraces", fetchMock as unknown as typeof fetch)).resolves.toBe("018a-task-1");
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe(MESHY_URL);
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer msy_secret_key");
    const body = JSON.parse(String(init.body));
    expect(body).toMatchObject({ mode: "preview", model_type: "standard", target_formats: ["glb"] });
    expect(body.prompt).toMatch(/a tower with terraces/);
  });
  it("refuses without a key and rejects responses without a task id", async () => {
    vi.stubEnv("MESHY_API_KEY", "");
    await expect(createMeshyPreview("x tower", "", vi.fn() as unknown as typeof fetch)).rejects.toMatchObject({ code: "not-configured" });
    vi.stubEnv("MESHY_API_KEY", "k");
    await expect(createMeshyPreview("x tower", "", (async () => json({})) as unknown as typeof fetch)).rejects.toMatchObject({ code: "bad-response" });
  });
});

describe("normalizeTask", () => {
  it.each([["PENDING", "queued"], ["IN_PROGRESS", "running"], ["CANCELED", "cancelled"]] as const)("maps %s → %s", (raw, status) => {
    expect(normalizeTask({ id: "t", status: raw, progress: 41.6 })).toMatchObject({ providerTaskId: "t", status, progress: 42 });
  });
  it("maps SUCCEEDED with a GLB URL and expiry", () => {
    expect(normalizeTask({ id: "t", status: "SUCCEEDED", progress: 100, model_urls: { glb: "https://assets.meshy.ai/a.glb", fbx: "x" }, expires_at: 1_800_000_000_000 })).toEqual({ providerTaskId: "t", status: "completed", progress: 100, glbUrl: "https://assets.meshy.ai/a.glb", expiresAt: new Date(1_800_000_000_000).toISOString() });
  });
  it("turns SUCCEEDED without a GLB, FAILED, unknown statuses, and garbage into failures — never success", () => {
    expect(normalizeTask({ id: "t", status: "SUCCEEDED", model_urls: {} })).toMatchObject({ status: "failed", error: { code: "no-glb" } });
    expect(normalizeTask({ id: "t", status: "FAILED", task_error: { message: "Prompt rejected" } })).toMatchObject({ status: "failed", error: { code: "provider-failed", message: "Prompt rejected", retryable: true } });
    expect(normalizeTask({ id: "t", status: "WEIRD" })).toMatchObject({ status: "failed", error: { code: "unknown-status" } });
    expect(normalizeTask("nope", "fallback")).toMatchObject({ providerTaskId: "fallback", status: "failed", error: { code: "bad-response" } });
  });
  it("clamps progress", () => expect(normalizeTask({ id: "t", status: "IN_PROGRESS", progress: 250 }).progress).toBe(100));
});

describe("error mapping", () => {
  it.each([[401, "auth", false], [402, "insufficient-credits", false], [404, "not-found", false], [409, "running", false], [429, "rate-limited", true], [500, "provider-unavailable", true], [503, "provider-unavailable", true]] as const)("HTTP %i → %s", (status, code, retryable) => {
    expect(mapHttpError(status)).toMatchObject({ code, retryable });
  });
  it("reads Retry-After", () => expect(mapHttpError(429, "17").retryAfterSeconds).toBe(17));
  it("surfaces provider errors from getMeshyTask and maps network/timeouts", async () => {
    await expect(getMeshyTask("task-12345", (async () => new Response("", { status: 429, headers: { "retry-after": "5" } })) as unknown as typeof fetch)).rejects.toMatchObject({ code: "rate-limited", retryAfterSeconds: 5 });
    await expect(getMeshyTask("task-12345", (async () => { throw new TypeError("fetch failed"); }) as unknown as typeof fetch)).rejects.toMatchObject({ code: "network", retryable: true });
    await expect(getMeshyTask("task-12345", (async () => { const e = new Error("t"); e.name = "TimeoutError"; throw e; }) as unknown as typeof fetch)).rejects.toMatchObject({ code: "timeout" });
  });
  it("never leaks the API key in error messages", async () => {
    const error = await getMeshyTask("task-12345", (async () => new Response("key msy_secret_key invalid", { status: 401 })) as unknown as typeof fetch).catch((e: MeshyError) => e);
    expect(String((error as MeshyError).message)).not.toContain("msy_secret_key");
  });
});

describe("getMeshyTask / deleteMeshyTask", () => {
  it("GETs the task by id and normalizes it", async () => {
    const fetchMock = vi.fn(async (..._args: unknown[]) => json({ id: "task-12345", status: "IN_PROGRESS", progress: 30 }));
    await expect(getMeshyTask("task-12345", fetchMock as unknown as typeof fetch)).resolves.toMatchObject({ status: "running", progress: 30 });
    expect(fetchMock.mock.calls[0][0]).toBe(`${MESHY_URL}/task-12345`);
  });
  it("DELETE succeeds for queued tasks and reports `running` on 409", async () => {
    const del = vi.fn(async (..._args: unknown[]) => new Response(null, { status: 200 }));
    await expect(deleteMeshyTask("task-12345", del as unknown as typeof fetch)).resolves.toBeUndefined();
    expect((del.mock.calls[0][1] as RequestInit).method).toBe("DELETE");
    await expect(deleteMeshyTask("task-12345", (async () => new Response("", { status: 409 })) as unknown as typeof fetch)).rejects.toMatchObject({ code: "running" });
  });
});

describe("downloadGlb", () => {
  it("only fetches HTTPS Meshy hosts", () => {
    expect(isAllowedAssetUrl("https://assets.meshy.ai/x.glb")).toBe(true);
    ["http://assets.meshy.ai/x.glb", "https://evil.com/x.glb", "https://meshy.ai.evil.com/x.glb", "https://169.254.169.254/latest", "file:///etc/passwd", "nonsense"].forEach((u) => expect(isAllowedAssetUrl(u)).toBe(false));
  });
  it("returns valid GLB bytes", async () => {
    const bytes = await downloadGlb("https://assets.meshy.ai/x.glb", (async () => new Response(glb())) as unknown as typeof fetch);
    expect(bytes.byteLength).toBe(32);
  });
  it("rejects disallowed hosts without making a request", async () => {
    const f = vi.fn();
    await expect(downloadGlb("https://evil.com/x.glb", f as unknown as typeof fetch)).rejects.toMatchObject({ code: "bad-asset-url" });
    expect(f).not.toHaveBeenCalled();
  });
  it("detects expired links, non-GLB payloads, and oversized files", async () => {
    await expect(downloadGlb("https://assets.meshy.ai/x.glb", (async () => new Response("", { status: 403 })) as unknown as typeof fetch)).rejects.toMatchObject({ code: "asset-expired", retryable: true });
    await expect(downloadGlb("https://assets.meshy.ai/x.glb", (async () => new Response("<html>not a model</html>")) as unknown as typeof fetch)).rejects.toMatchObject({ code: "not-glb" });
    await expect(downloadGlb("https://assets.meshy.ai/x.glb", (async () => new Response(glb(), { headers: { "content-length": String(500 * 1024 * 1024) } })) as unknown as typeof fetch)).rejects.toMatchObject({ code: "too-large" });
  });
});

describe("paid-request guard", () => {
  const env = { MESHY_ENABLED: "true", MESHY_API_KEY: "k", MESHY_ACCESS_CODE: "letmein" };
  it("fails closed when not fully configured", () => {
    expect(authorize(new Headers(), { ...env, MESHY_ACCESS_CODE: undefined })).toMatchObject({ ok: false, status: 503 });
    expect(authorize(new Headers(), { ...env, MESHY_ENABLED: "false" })).toMatchObject({ ok: false, status: 503 });
    expect(authorize(new Headers(), { ...env, MESHY_API_KEY: "" })).toMatchObject({ ok: false, status: 503 });
  });
  it("requires the exact access code", () => {
    expect(authorize(new Headers(), env)).toMatchObject({ ok: false, status: 401 });
    expect(authorize(new Headers({ "x-sift-access-code": "wrong" }), env)).toMatchObject({ ok: false, status: 401 });
    expect(authorize(new Headers({ "x-sift-access-code": "letmein" }), env)).toEqual({ ok: true });
    expect(codeMatches("letmein", "letmein")).toBe(true);
    expect(codeMatches("letmeinX", "letmein")).toBe(false);
    expect(codeMatches("", "")).toBe(false);
  });
  it("limits per IP over a window and per day overall, and only counts recorded spends", () => {
    const l = new SpendLimiter({ perIpWindowMs: 60_000, perIpMax: 2, dailyMax: 3 });
    expect(l.check("a", 0)).toEqual({ ok: true });
    expect(l.check("a", 1)).toEqual({ ok: true }); // checking does not consume
    l.record("a", 1); l.record("a", 2);
    expect(l.check("a", 3)).toMatchObject({ ok: false, status: 429, code: "rate-limited" });
    expect(l.check("b", 3)).toEqual({ ok: true });
    expect(l.check("a", 61_000)).toEqual({ ok: true }); // window passed
    l.record("b", 61_000);
    expect(l.check("c", 62_000)).toMatchObject({ ok: false, code: "daily-limit" });
    expect(l.check("c", 24 * 3600_000 + 5)).toEqual({ ok: true });
  });
});

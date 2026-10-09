import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { downloadGlbFor } from "./http";
import { TRIPO_BASE_URL, TRIPO_MODEL, tripoProvider, truncatePrompt } from "./tripo";

// NOTE: documented-contract tests against a mocked fetch (v3 contract reconstructed from secondary sources, UNVERIFIED). They do not prove live Tripo behaviour.
const json = (body: unknown, init: ResponseInit = {}) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" }, ...init });
const glb = (extra = 20) => { const b = new Uint8Array(12 + extra); b.set([0x67, 0x6c, 0x54, 0x46]); return b.buffer; };
const asFetch = (f: unknown) => f as typeof fetch;
const ID = "1ec04ced-4b87-40f6-a2c4-8d1a1e0b3f55";
const task = (data: Record<string, unknown>) => json({ code: 0, data: { task_id: ID, ...data } });
const statusOf = (data: Record<string, unknown>) => tripoProvider.status(ID, asFetch(async () => task(data)));

beforeEach(() => { vi.stubEnv("TRIPO_ENABLED", "true"); vi.stubEnv("TRIPO_API_KEY", "tr_secret_key"); vi.stubEnv("SIFT_ACCESS_CODE", "letmein"); });
afterEach(() => vi.unstubAllEnvs());

describe("config", () => {
  const base = { TRIPO_ENABLED: "true", TRIPO_API_KEY: "k", SIFT_ACCESS_CODE: "c" };
  it("requires flag, key, and an access code", () => {
    expect(tripoProvider.config(base)).toMatchObject({ configured: true, verified: false });
    expect(tripoProvider.config({ ...base, TRIPO_ENABLED: "false" }).configured).toBe(false);
    expect(tripoProvider.config({ ...base, TRIPO_API_KEY: "" })).toMatchObject({ configured: false, hasKey: false });
    expect(tripoProvider.config({ ...base, SIFT_ACCESS_CODE: undefined })).toMatchObject({ configured: false, accessCodeRequired: false });
  });
  it("ignores the removed MESHY_ACCESS_CODE fallback", () => {
    expect(tripoProvider.config({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "k", MESHY_ACCESS_CODE: "m" }).configured).toBe(false);
  });
  it("reports static metadata", () => {
    expect(tripoProvider).toMatchObject({ id: "tripo", label: "Tripo", supportsCancel: false, assetHosts: ["tripo3d.com", "tripo3d.ai"] });
    expect(tripoProvider.costLabel).toBe("≈ $0.30 per model (estimate)");
  });
});

describe("create", () => {
  it("POSTs v3 /generation/text-to-model with bearer key and returns the task id", async () => {
    const f = vi.fn(async (..._a: unknown[]) => json({ code: 0, data: { task_id: ID } }));
    await expect(tripoProvider.create("A tower", "with terraces", asFetch(f))).resolves.toBe(ID);
    const [url, init] = f.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("https://openapi.tripo3d.ai/v3/generation/text-to-model");
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer tr_secret_key");
    const body = JSON.parse(String(init.body));
    expect(body).toMatchObject({ model: TRIPO_MODEL, texture: false, pbr: false });
    expect(body).not.toHaveProperty("type");
    expect(body.prompt).toMatch(/a tower with terraces/);
    expect(body.prompt.length).toBeLessThanOrEqual(1024);
  });
  it("refuses without a key and never calls the network", async () => {
    vi.stubEnv("TRIPO_API_KEY", "");
    const f = vi.fn();
    await expect(tripoProvider.create("x tower", "", asFetch(f))).rejects.toMatchObject({ code: "not-configured" });
    expect(f).not.toHaveBeenCalled();
  });
  it("rejects missing or malformed task ids", async () => {
    await expect(tripoProvider.create("x tower", "", asFetch(async () => json({ code: 0, data: {} })))).rejects.toMatchObject({ code: "bad-response" });
    await expect(tripoProvider.create("x tower", "", asFetch(async () => json({ code: 0, data: { task_id: "../etc/passwd" } })))).rejects.toMatchObject({ code: "bad-response" });
    await expect(tripoProvider.create("x tower", "", asFetch(async () => json("garbage")))).rejects.toMatchObject({ code: "bad-response" });
  });
  it("truncates long prompts without splitting surrogate pairs", () => {
    expect(truncatePrompt("a".repeat(2000)).length).toBe(1024);
    const t = truncatePrompt("a".repeat(1023) + "😀");
    expect(t.length).toBe(1023);
  });
});

describe("status mapping", () => {
  it.each([["queued", "queued"], ["running", "running"], ["cancelled", "cancelled"]] as const)("maps %s", async (raw, expected) => {
    await expect(statusOf({ status: raw, progress: 41.6 })).resolves.toMatchObject({ providerTaskId: ID, status: expected, progress: 42 });
  });
  it("requests GET /v3/tasks/{id}", async () => {
    const f = vi.fn(async (..._a: unknown[]) => task({ status: "running" }));
    await tripoProvider.status(ID, asFetch(f));
    expect(f.mock.calls[0][0]).toBe(`${TRIPO_BASE_URL}/tasks/${ID}`);
    expect((f.mock.calls[0][1] as RequestInit).method).toBe("GET");
  });
  it("success yields glbUrl from each output shape, preferring v3 model_url", async () => {
    const u = (n: string) => `https://tripo-data.rg1.data.tripo3d.com/${n}.glb`;
    await expect(statusOf({ status: "success", progress: 100, output: { model_url: u("mu"), pbr_model: u("pbr"), model: u("m") } })).resolves.toMatchObject({ status: "completed", glbUrl: u("mu") });
    await expect(statusOf({ status: "success", output: { pbr_model: u("pbr"), model: u("m") } })).resolves.toMatchObject({ glbUrl: u("pbr") });
    await expect(statusOf({ status: "success", output: { model: u("m") } })).resolves.toMatchObject({ status: "completed", glbUrl: u("m"), progress: 100 });
    await expect(statusOf({ status: "success", output: { model_url: u("mu") } })).resolves.toMatchObject({ glbUrl: u("mu") });
    await expect(statusOf({ status: "success", output: { pbr_model: { url: u("obj") } } })).resolves.toMatchObject({ glbUrl: u("obj") });
  });
  it("parses an Expires epoch-seconds query param only when present", async () => {
    await expect(statusOf({ status: "success", output: { model: "https://x.tripo3d.com/a.glb?Expires=1800000000&Sig=z" } })).resolves.toMatchObject({ expiresAt: new Date(1_800_000_000_000).toISOString() });
    expect(await statusOf({ status: "success", output: { model: "https://x.tripo3d.com/a.glb" } })).not.toHaveProperty("expiresAt");
  });
  it("success without a model URL fails with no-glb", async () => {
    await expect(statusOf({ status: "success", output: {} })).resolves.toMatchObject({ status: "failed", error: { code: "no-glb", retryable: false } });
    await expect(statusOf({ status: "success" })).resolves.toMatchObject({ status: "failed", error: { code: "no-glb" } });
  });
  it("failed/banned/expired/unknown/garbage never succeed", async () => {
    await expect(statusOf({ status: "failed" })).resolves.toMatchObject({ status: "failed", error: { code: "provider-failed" } });
    await expect(statusOf({ status: "banned" })).resolves.toMatchObject({ status: "failed", error: { code: "banned", retryable: false } });
    await expect(statusOf({ status: "expired" })).resolves.toMatchObject({ status: "failed", error: { code: "expired", retryable: false } });
    await expect(statusOf({ status: "unknown" })).resolves.toMatchObject({ status: "failed", error: { code: "unknown-status" } });
    await expect(statusOf({ status: "weird" })).resolves.toMatchObject({ status: "failed", error: { code: "unknown-status" } });
    await expect(statusOf({})).resolves.toMatchObject({ status: "failed", error: { code: "unknown-status" } });
    await expect(tripoProvider.status(ID, asFetch(async () => json({ code: 0 })))).resolves.toMatchObject({ status: "failed", error: { code: "bad-response" } });
  });
  it("clamps progress and rejects bad task ids without a request", async () => {
    expect((await statusOf({ status: "running", progress: 250 })).progress).toBe(100);
    const f = vi.fn();
    await expect(tripoProvider.status("a/b", asFetch(f))).rejects.toMatchObject({ code: "bad-response" });
    await expect(tripoProvider.status("short", asFetch(f))).rejects.toMatchObject({ code: "bad-response" });
    expect(f).not.toHaveBeenCalled();
  });
});

describe("errors", () => {
  it.each([[402, "insufficient-credits", false], [429, "rate-limited", true], [403, "auth", false], [401, "auth", false], [500, "provider-unavailable", true], [503, "provider-unavailable", true]] as const)("HTTP %i -> %s", async (s, code, retryable) => {
    await expect(tripoProvider.status(ID, asFetch(async () => new Response("", { status: s })))).rejects.toMatchObject({ code, retryable });
  });
  it("reads Retry-After on 429", async () => {
    await expect(tripoProvider.status(ID, asFetch(async () => new Response("", { status: 429, headers: { "retry-after": "7" } })))).rejects.toMatchObject({ retryAfterSeconds: 7 });
  });
  it("maps HTTP 403 with envelope code 2010 to insufficient-credits, plain 403 to auth", async () => {
    await expect(tripoProvider.create("x tower", "", asFetch(async () => json({ code: 2010, message: "no credit" }, { status: 403 })))).rejects.toMatchObject({ code: "insufficient-credits", retryable: false });
    await expect(tripoProvider.create("x tower", "", asFetch(async () => new Response("nope", { status: 403 })))).rejects.toMatchObject({ code: "auth" });
  });
  it("maps network failures and timeouts", async () => {
    await expect(tripoProvider.status(ID, asFetch(async () => { throw new TypeError("fetch failed"); }))).rejects.toMatchObject({ code: "network", retryable: true });
    await expect(tripoProvider.status(ID, asFetch(async () => { const e = new Error("t"); e.name = "TimeoutError"; throw e; }))).rejects.toMatchObject({ code: "timeout", retryable: true });
  });
  it("maps non-zero envelope codes", async () => {
    const env = (code: number) => asFetch(async () => json({ code, message: "x" }));
    await expect(tripoProvider.create("x tower", "", env(2010))).rejects.toMatchObject({ code: "insufficient-credits", retryable: false });
    await expect(tripoProvider.create("x tower", "", env(2000))).rejects.toMatchObject({ code: "rate-limited", retryable: true });
    await expect(tripoProvider.status(ID, env(1234))).rejects.toMatchObject({ code: "provider-error", retryable: false });
  });
  it("never leaks the API key in error messages", async () => {
    const bodies = [new Response("key tr_secret_key invalid", { status: 401 }), json({ code: 9, message: "bad key tr_secret_key" })];
    for (const r of bodies) {
      const e = await tripoProvider.status(ID, asFetch(async () => r)).catch((x: Error) => x);
      expect(String((e as Error).message)).not.toContain("tr_secret_key");
    }
  });
});

describe("cancel", () => {
  it("is unsupported and never touches the network", async () => {
    const f = vi.fn();
    await expect(tripoProvider.cancel(ID, asFetch(f))).rejects.toMatchObject({ code: "cancel-unsupported", httpStatus: 501, retryable: false });
    expect(f).not.toHaveBeenCalled();
  });
});

describe("downloadGlbFor(tripoProvider)", () => {
  const url = "https://tripo-data.rg1.data.tripo3d.com/m.glb";
  it("downloads valid GLBs from allowlisted hosts", async () => {
    expect((await downloadGlbFor(tripoProvider, url, asFetch(async () => new Response(glb())))).byteLength).toBe(32);
  });
  it("rejects bad hosts without a request", async () => {
    const f = vi.fn();
    for (const u of ["https://evil.com/m.glb", "https://tripo3d.com.evil.com/m.glb", "http://x.tripo3d.com/m.glb"]) await expect(downloadGlbFor(tripoProvider, u, asFetch(f))).rejects.toMatchObject({ code: "bad-asset-url" });
    expect(f).not.toHaveBeenCalled();
  });
  it("enforces size cap and GLB magic", async () => {
    await expect(downloadGlbFor(tripoProvider, url, asFetch(async () => new Response(glb(), { headers: { "content-length": String(500 * 1024 * 1024) } })))).rejects.toMatchObject({ code: "too-large" });
    await expect(downloadGlbFor(tripoProvider, url, asFetch(async () => new Response("<html>nope</html>")))).rejects.toMatchObject({ code: "not-glb" });
  });
});

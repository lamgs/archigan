import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { downloadGlbFor } from "./http";
import { falAppId, hunyuanProProvider, hunyuanRapidProvider, truncateUtf8 } from "./hunyuan";
import type { HostedProvider } from "./types";

// NOTE: documented-contract tests against a mocked fetch. The fal.ai contract is UNVERIFIED; nothing here proves live behaviour.
const KEY = "fal_secret_key_123";
const json = (body: unknown, init: ResponseInit = {}) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" }, ...init });
const glb = (extra = 20) => { const b = new Uint8Array(12 + extra); b.set([0x67, 0x6c, 0x54, 0x46]); return b.buffer; };
const mock = (impl: (url: string, init: RequestInit) => Response | Promise<Response>) => vi.fn(async (url: unknown, init: unknown) => impl(String(url), init as RequestInit)) as unknown as typeof fetch & ReturnType<typeof vi.fn>;
const calls = (f: unknown) => (f as ReturnType<typeof vi.fn>).mock.calls as [string, RequestInit][];
const ID = "0a1b2c3d-4e5f-6789";

const tiers: [string, HostedProvider, string, number][] = [
  ["rapid", hunyuanRapidProvider, "fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d", 200],
  ["pro", hunyuanProProvider, "fal-ai/hunyuan-3d/v3.1/pro/text-to-3d", 1024],
];

beforeEach(() => { vi.stubEnv("HUNYUAN_ENABLED", "true"); vi.stubEnv("FAL_KEY", KEY); vi.stubEnv("SIFT_ACCESS_CODE", "letmein"); });
afterEach(() => vi.unstubAllEnvs());

describe("metadata", () => {
  it("exposes ids, labels, costs, hosts", () => {
    expect(hunyuanRapidProvider).toMatchObject({ id: "hunyuan3d-rapid", label: "Hunyuan3D Rapid", supportsCancel: true, assetHosts: ["fal.media"] });
    expect(hunyuanProProvider).toMatchObject({ id: "hunyuan3d-pro", label: "Hunyuan3D Pro", supportsCancel: true });
    expect(hunyuanRapidProvider.costLabel).toContain("0.225");
    expect(hunyuanProProvider.costLabel).toContain("0.375");
  });
  it("derives the app id from the first two endpoint segments", () => expect(falAppId(tiers[0][2])).toBe("fal-ai/hunyuan-3d"));
});

describe.each(tiers)("%s create", (_n, provider, endpoint, limit) => {
  it("POSTs to the queue endpoint with a Key header and returns request_id", async () => {
    const f = mock(() => json({ request_id: ID, status_url: "x", response_url: "y", cancel_url: "z" }));
    await expect(provider.create("A tower", "with terraces", f)).resolves.toBe(ID);
    const [url, init] = calls(f)[0];
    expect(url).toBe(`https://queue.fal.run/${endpoint}`);
    expect(init.method).toBe("POST");
    expect((init.headers as Record<string, string>).Authorization).toBe(`Key ${KEY}`);
    const body = JSON.parse(String(init.body));
    expect(body.prompt).toMatch(/a tower with terraces/);
    expect(new TextEncoder().encode(body.prompt).length).toBeLessThanOrEqual(limit);
  });
  it("keeps the prompt within the limit for long and multibyte briefs", async () => {
    for (const brief of ["x".repeat(3000), "塔".repeat(3000), "tower 🏗 ".repeat(300)]) {
      const f = mock(() => json({ request_id: ID }));
      await provider.create(brief, "", f);
      const prompt = JSON.parse(String(calls(f)[0][1].body)).prompt as string;
      expect(new TextEncoder().encode(prompt).length).toBeLessThanOrEqual(limit);
      expect(prompt.length).toBeGreaterThan(0);
      expect(prompt).not.toContain("�");
    }
  });
  it("rejects missing or malformed request ids with bad-response", async () => {
    for (const body of [{}, { request_id: "short" }, { request_id: "bad id!!" }, { request_id: 12345678 }]) {
      await expect(provider.create("a tower", "", mock(() => json(body)))).rejects.toMatchObject({ code: "bad-response" });
    }
  });
  it("refuses without a key", async () => {
    vi.stubEnv("FAL_KEY", "");
    const f = mock(() => json({}));
    await expect(provider.create("a tower", "", f)).rejects.toMatchObject({ code: "not-configured" });
    expect(calls(f)).toHaveLength(0);
  });
});

it("sends tier-specific options (UNVERIFIED names)", async () => {
  const f1 = mock(() => json({ request_id: ID })); await hunyuanRapidProvider.create("a tower", "", f1);
  expect(JSON.parse(String(calls(f1)[0][1].body))).toHaveProperty("enable_pbr");
  const f2 = mock(() => json({ request_id: ID })); await hunyuanProProvider.create("a tower", "", f2);
  const face = JSON.parse(String(calls(f2)[0][1].body)).face_count;
  expect(face).toBeGreaterThanOrEqual(40_000); expect(face).toBeLessThanOrEqual(1_500_000);
});

it("truncateUtf8 never splits a code point", () => {
  expect(truncateUtf8("ab🏗cd", 5)).toBe("ab");
  expect(truncateUtf8("abc", 10)).toBe("abc");
});

describe.each(tiers)("%s status", (_n, provider, endpoint) => {
  const app = "https://queue.fal.run/fal-ai/hunyuan-3d/requests";
  it.each([["IN_QUEUE", "queued"], ["IN_PROGRESS", "running"]] as const)("maps %s → %s without progress", async (raw, status) => {
    const f = mock(() => json({ status: raw, queue_position: 2, logs: [] }));
    const task = await provider.status(ID, f);
    expect(task).toEqual({ providerTaskId: ID, status });
    expect(calls(f)[0][0]).toBe(`${app}/${ID}/status`);
    expect((calls(f)[0][1].headers as Record<string, string>).Authorization).toBe(`Key ${KEY}`);
    expect(endpoint).toContain("fal-ai/hunyuan-3d");
  });
  it("COMPLETED fetches the result and returns the glb url", async () => {
    const f = mock((url) => url.endsWith("/status") ? json({ status: "COMPLETED" }) : json({ model_glb: { url: "https://v3.fal.media/files/a/model.glb", content_type: "model/gltf-binary", file_name: "model.glb" } }));
    await expect(provider.status(ID, f)).resolves.toEqual({ providerTaskId: ID, status: "completed", glbUrl: "https://v3.fal.media/files/a/model.glb" });
    expect(calls(f)[1][0]).toBe(`${app}/${ID}`);
  });
  it("tolerates model_urls.glb.url", async () => {
    const f = mock((url) => url.endsWith("/status") ? json({ status: "COMPLETED" }) : json({ model_urls: { glb: { url: "https://fal.media/a.glb" } } }));
    await expect(provider.status(ID, f)).resolves.toMatchObject({ status: "completed", glbUrl: "https://fal.media/a.glb" });
  });
  it("fails on an error/detail in the result, a 422 result, or a status error", async () => {
    for (const result of [{ detail: [{ msg: "bad" }] }, { error: "boom" }]) {
      const f = mock((url) => url.endsWith("/status") ? json({ status: "COMPLETED" }) : json(result));
      await expect(provider.status(ID, f)).resolves.toMatchObject({ status: "failed", error: { code: "provider-failed" } });
    }
    const f422 = mock((url) => url.endsWith("/status") ? json({ status: "COMPLETED" }) : json({ detail: "x" }, { status: 422 }));
    await expect(provider.status(ID, f422)).resolves.toMatchObject({ status: "failed", error: { code: "provider-failed" } });
    await expect(provider.status(ID, mock(() => json({ status: "COMPLETED", error: "x" })))).resolves.toMatchObject({ status: "failed" });
  });
  it("fails a .obj model_glb with not-glb, and a missing file with no-glb", async () => {
    const obj = mock((url) => url.endsWith("/status") ? json({ status: "COMPLETED" }) : json({ model_glb: { url: "https://fal.media/m.obj", content_type: "text/plain", file_name: "m.obj" } }));
    await expect(provider.status(ID, obj)).resolves.toMatchObject({ status: "failed", error: { code: "not-glb" } });
    const none = mock((url) => url.endsWith("/status") ? json({ status: "COMPLETED" }) : json({}));
    await expect(provider.status(ID, none)).resolves.toMatchObject({ status: "failed", error: { code: "no-glb" } });
  });
  it("treats unknown statuses and garbage as failures, never success", async () => {
    await expect(provider.status(ID, mock(() => json({ status: "WEIRD" })))).resolves.toMatchObject({ status: "failed", error: { code: "unknown-status" } });
    await expect(provider.status(ID, mock(() => json("nope")))).resolves.toMatchObject({ status: "failed", error: { code: "bad-response" } });
  });
  it("rejects ids that could alter the request path", async () => {
    const f = mock(() => json({}));
    await expect(provider.status("../../x", f)).rejects.toMatchObject({ code: "bad-task-id" });
    expect(calls(f)).toHaveLength(0);
  });
});

describe("errors", () => {
  const p = hunyuanRapidProvider;
  it.each([[401, "auth"], [402, "insufficient-credits"], [403, "auth"], [404, "not-found"], [429, "rate-limited"], [500, "provider-unavailable"], [503, "provider-unavailable"]] as const)("HTTP %i → %s", async (status, code) => {
    await expect(p.create("a tower", "", mock(() => new Response("{}", { status })))).rejects.toMatchObject({ code });
    await expect(p.status(ID, mock(() => new Response("{}", { status })))).rejects.toMatchObject({ code });
  });
  it("honours Retry-After on 429", async () => {
    await expect(p.create("a tower", "", mock(() => new Response("{}", { status: 429, headers: { "retry-after": "30" } })))).rejects.toMatchObject({ retryAfterSeconds: 30, retryable: true });
  });
  it("maps 403 with an exhausted-balance body to insufficient-credits", async () => {
    await expect(p.create("a tower", "", mock(() => new Response('{"detail":"User is locked. Reason: Exhausted balance."}', { status: 403 })))).rejects.toMatchObject({ code: "insufficient-credits", retryable: false });
  });
  it("maps timeouts and network failures", async () => {
    const timeout = mock(() => { throw new DOMException("t", "TimeoutError"); });
    await expect(p.create("a tower", "", timeout)).rejects.toMatchObject({ code: "timeout", retryable: true });
    await expect(p.status(ID, mock(() => { throw new TypeError("fetch failed"); }))).rejects.toMatchObject({ code: "network" });
  });
  it("never leaks the key or response bodies in thrown messages", async () => {
    const leaky = mock(() => new Response(`oops ${KEY}`, { status: 403 }));
    const errs: unknown[] = [];
    for (const run of [() => p.create("a tower", "", leaky), () => p.status(ID, leaky), () => p.cancel(ID, leaky), () => p.create("a tower", "", mock(() => { throw new Error(`conn ${KEY}`); }))]) {
      errs.push(await run().catch((e) => e));
    }
    for (const e of errs) { expect(e).toBeInstanceOf(Error); expect(JSON.stringify({ m: (e as Error).message, ...(e as object) })).not.toContain(KEY); }
  });
});

describe("cancel", () => {
  it("PUTs the cancel route", async () => {
    const f = mock(() => json({ status: "CANCELLATION_REQUESTED" }, { status: 202 }));
    await expect(hunyuanProProvider.cancel(ID, f)).resolves.toBeUndefined();
    const [url, init] = calls(f)[0];
    expect(url).toBe(`https://queue.fal.run/fal-ai/hunyuan-3d/requests/${ID}/cancel`);
    expect(init.method).toBe("PUT");
  });
  it("reports already-completed on 400", async () => {
    await expect(hunyuanRapidProvider.cancel(ID, mock(() => json({ status: "ALREADY_COMPLETED" }, { status: 400 })))).rejects.toMatchObject({ code: "already-completed", retryable: false });
  });
  it("surfaces other failures", async () => {
    await expect(hunyuanRapidProvider.cancel(ID, mock(() => new Response("{}", { status: 404 })))).rejects.toMatchObject({ code: "not-found" });
  });
});

describe("config (fail closed)", () => {
  const base = { HUNYUAN_ENABLED: "true", FAL_KEY: "k", SIFT_ACCESS_CODE: "c" };
  it.each([hunyuanRapidProvider, hunyuanProProvider])("$id requires flag, key, and an access code", (p) => {
    expect(p.config(base)).toMatchObject({ configured: true, verified: false });
    expect(p.config({ ...base, HUNYUAN_ENABLED: "false" }).configured).toBe(false);
    expect(p.config({ ...base, HUNYUAN_ENABLED: undefined }).configured).toBe(false);
    expect(p.config({ ...base, FAL_KEY: "" })).toMatchObject({ configured: false, hasKey: false });
    expect(p.config({ ...base, SIFT_ACCESS_CODE: undefined })).toMatchObject({ configured: false, accessCodeRequired: false });
    expect(p.config({ HUNYUAN_ENABLED: "true", FAL_KEY: "k", MESHY_ACCESS_CODE: "m" }).configured).toBe(true);
    expect(p.config({ HUNYUAN_ENABLED: "true", MESHY_ACCESS_CODE: "m" }).configured).toBe(false);
  });
});

describe("downloadGlbFor", () => {
  const p = hunyuanRapidProvider;
  const ok = "https://v3.fal.media/files/a/model.glb";
  it("downloads a GLB from an allowlisted host", async () => {
    await expect(downloadGlbFor(p, ok, mock(() => new Response(glb())))).resolves.toBeInstanceOf(ArrayBuffer);
  });
  it("rejects foreign hosts, http, lookalikes, and never fetches", async () => {
    const f = mock(() => new Response(glb()));
    for (const url of ["https://evil.example/m.glb", "http://v3.fal.media/m.glb", "https://notfal.media/m.glb", "https://fal.media.evil.com/m.glb"]) {
      await expect(downloadGlbFor(p, url, f)).rejects.toMatchObject({ code: "bad-asset-url" });
    }
    expect(calls(f)).toHaveLength(0);
  });
  it("enforces the size cap and the GLB magic number", async () => {
    await expect(downloadGlbFor({ ...p, maxGlbBytes: 16 }, ok, mock(() => new Response(glb(100))))).rejects.toMatchObject({ code: "too-large" });
    await expect(downloadGlbFor(p, ok, mock(() => new Response("<html>not a model</html>")))).rejects.toMatchObject({ code: "not-glb" });
  });
});

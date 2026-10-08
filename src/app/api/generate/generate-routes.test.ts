import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetSpendLimiterForTests } from "@/lib/providers/hosted-http";
import { DELETE, GET } from "./[taskId]/route";
import { GET as GET_MODEL } from "./[taskId]/model/route";
import { POST } from "./route";

// Documented-contract route tests with a mocked global fetch (no live Meshy calls).
const post = (body: unknown, headers: Record<string, string> = {}) => new Request("http://localhost/api/generate", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });
const ctx = (taskId: string) => ({ params: Promise.resolve({ taskId }) });
const authed = { "x-sift-access-code": "letmein", "x-forwarded-for": "1.2.3.4" };
const meshy = { prompt: "A terraced tower", refinement: "", provider: "meshy", confirmSpend: true };
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  resetSpendLimiterForTests();
  vi.stubEnv("MESHY_ENABLED", "true"); vi.stubEnv("MESHY_API_KEY", "msy_secret_key"); vi.stubEnv("MESHY_ACCESS_CODE", "letmein"); vi.stubEnv("MESHY_DAILY_LIMIT", "20");
  fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
});
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); });

describe("POST /api/generate", () => {
  it("keeps the procedural path free: no key, no code, no network", async () => {
    vi.unstubAllEnvs();
    const res = await POST(post({ prompt: "A library", refinement: "", provider: "procedural" }));
    expect(res.status).toBe(200);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("rejects hosted requests when unconfigured, without the code, or without confirmation — and never calls Meshy", async () => {
    vi.stubEnv("MESHY_ACCESS_CODE", "");
    expect((await POST(post(meshy, authed))).status).toBe(503);
    vi.stubEnv("MESHY_ACCESS_CODE", "letmein");
    expect((await POST(post(meshy))).status).toBe(401);
    expect((await POST(post(meshy, { "x-sift-access-code": "nope" }))).status).toBe(401);
    const unconfirmed = await POST(post({ ...meshy, confirmSpend: undefined }, authed));
    expect(unconfirmed.status).toBe(400);
    expect(await unconfirmed.json()).toMatchObject({ code: "confirmation-required" });
    expect((await POST(post({ ...meshy, confirmSpend: "yes" }, authed))).status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("creates a task when authorized and confirmed, and never exposes the key", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ result: "task-abc-123" }), { status: 200 }));
    const res = await POST(post(meshy, authed));
    expect(res.status).toBe(202);
    const text = await res.text();
    expect(JSON.parse(text)).toMatchObject({ kind: "task", taskId: "task-abc-123", status: "queued", verified: false });
    expect(text).not.toContain("msy_secret_key");
  });
  it("rate limits per IP, but failed provider calls do not consume the budget", async () => {
    fetchMock.mockResolvedValueOnce(new Response("", { status: 402 }));
    const failed = await POST(post(meshy, authed));
    expect(failed.status).toBe(402);
    expect(await failed.json()).toMatchObject({ code: "insufficient-credits", retryable: false });
    fetchMock.mockImplementation(async () => new Response(JSON.stringify({ result: "task-abc-123" }), { status: 200 }));
    for (let i = 0; i < 3; i++) expect((await POST(post(meshy, authed))).status).toBe(202);
    const limited = await POST(post(meshy, authed));
    expect(limited.status).toBe(429);
    expect(limited.headers.get("retry-after")).toBeTruthy();
  });
  it("relays provider rate limiting with Retry-After", async () => {
    fetchMock.mockImplementation(async () => new Response("", { status: 429, headers: { "retry-after": "9" } }));
    const res = await POST(post(meshy, authed));
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("9");
    expect(await res.json()).toMatchObject({ code: "rate-limited", retryable: true });
  });
});

describe("GET/DELETE /api/generate/[taskId]", () => {
  const get = (code = "letmein") => new Request("http://localhost/api/generate/task-abc-123", { headers: { "x-sift-access-code": code } });
  it("requires the access code and a well-formed id", async () => {
    expect((await GET(get("wrong"), ctx("task-abc-123"))).status).toBe(401);
    expect((await GET(get(), ctx("../../etc"))).status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("returns normalized status without the signed model URL", async () => {
    fetchMock.mockResolvedValue(new Response(JSON.stringify({ id: "task-abc-123", status: "SUCCEEDED", progress: 100, model_urls: { glb: "https://assets.meshy.ai/signed.glb?sig=secret" } }), { status: 200 }));
    const res = await GET(get(), ctx("task-abc-123"));
    const text = await res.text();
    expect(JSON.parse(text).task).toMatchObject({ status: "completed", hasModel: true });
    expect(text).not.toContain("signed.glb");
  });
  it("cancels queued tasks and reports running tasks honestly", async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 200 }));
    expect(await (await DELETE(get(), ctx("task-abc-123"))).json()).toMatchObject({ ok: true, status: "cancelled" });
    fetchMock.mockResolvedValueOnce(new Response("", { status: 409 }));
    const running = await DELETE(get(), ctx("task-abc-123"));
    expect(running.status).toBe(409);
    expect(await running.json()).toMatchObject({ code: "running" });
  });
});

describe("GET /api/generate/[taskId]/model", () => {
  const req = () => new Request("http://localhost/x", { headers: { "x-sift-access-code": "letmein" } });
  const glb = () => { const b = new Uint8Array(40); b.set([0x67, 0x6c, 0x54, 0x46]); return b; };
  it("re-reads the task for a fresh URL and streams a valid GLB", async () => {
    fetchMock
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "task-abc-123", status: "SUCCEEDED", model_urls: { glb: "https://assets.meshy.ai/fresh.glb" } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(glb()));
    const res = await GET_MODEL(req(), ctx("task-abc-123"));
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("model/gltf-binary");
    expect((await res.arrayBuffer()).byteLength).toBe(40);
    expect(fetchMock.mock.calls[1][0]).toBe("https://assets.meshy.ai/fresh.glb");
  });
  it("refuses unfinished tasks and off-allowlist URLs", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ id: "task-abc-123", status: "IN_PROGRESS" }), { status: 200 }));
    expect((await GET_MODEL(req(), ctx("task-abc-123"))).status).toBe(409);
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({ id: "task-abc-123", status: "SUCCEEDED", model_urls: { glb: "https://evil.example/x.glb" } }), { status: 200 }));
    const res = await GET_MODEL(req(), ctx("task-abc-123"));
    expect(res.status).toBe(502);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

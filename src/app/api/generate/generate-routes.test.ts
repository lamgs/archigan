import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetSpendLimiterForTests } from "@/lib/providers/hosted-http";
import { DELETE, GET } from "./[taskId]/route";
import { GET as GET_MODEL } from "./[taskId]/model/route";
import { POST } from "./route";

// Documented-contract route tests with a mocked global fetch (no live Tripo calls).
const post = (body: unknown, headers: Record<string, string> = {}) => new Request("http://localhost/api/generate", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });
const ctx = (taskId: string) => ({ params: Promise.resolve({ taskId }) });
const authed = { "x-sift-access-code": "letmein", "x-forwarded-for": "1.2.3.4" };
const tripo = { prompt: "A terraced tower", refinement: "", provider: "tripo", confirmSpend: true };
const ok = (data: unknown) => new Response(JSON.stringify({ code: 0, data }), { status: 200 });
const task = (status: string, extra: object = {}) => ok({ task_id: "task-abc-123", status, ...extra });
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  resetSpendLimiterForTests();
  vi.stubEnv("TRIPO_ENABLED", "true"); vi.stubEnv("TRIPO_API_KEY", "tripo_secret_key"); vi.stubEnv("SIFT_ACCESS_CODE", "letmein"); vi.stubEnv("SIFT_DAILY_LIMIT", "20");
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
  it("rejects hosted requests when unconfigured, without the code, or without confirmation — and never calls Tripo", async () => {
    vi.stubEnv("SIFT_ACCESS_CODE", "");
    expect((await POST(post(tripo, authed))).status).toBe(503);
    vi.stubEnv("SIFT_ACCESS_CODE", "letmein");
    expect((await POST(post(tripo))).status).toBe(401);
    expect((await POST(post(tripo, { "x-sift-access-code": "nope" }))).status).toBe(401);
    const unconfirmed = await POST(post({ ...tripo, confirmSpend: undefined }, authed));
    expect(unconfirmed.status).toBe(400);
    expect(await unconfirmed.json()).toMatchObject({ code: "confirmation-required" });
    expect((await POST(post({ ...tripo, confirmSpend: "yes" }, authed))).status).toBe(400);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("creates a task when authorized and confirmed, and never exposes the key", async () => {
    fetchMock.mockResolvedValue(ok({ task_id: "task-abc-123" }));
    const res = await POST(post(tripo, authed));
    expect(res.status).toBe(202);
    const text = await res.text();
    expect(JSON.parse(text)).toMatchObject({ kind: "task", taskId: "task-abc-123", status: "queued", verified: false });
    expect(text).not.toContain("tripo_secret_key");
  });
  it("rate limits per IP, but failed provider calls do not consume the budget", async () => {
    fetchMock.mockResolvedValueOnce(new Response("", { status: 402 }));
    const failed = await POST(post(tripo, authed));
    expect(failed.status).toBe(402);
    expect(await failed.json()).toMatchObject({ code: "insufficient-credits", retryable: false });
    fetchMock.mockImplementation(async () => ok({ task_id: "task-abc-123" }));
    for (let i = 0; i < 3; i++) expect((await POST(post(tripo, authed))).status).toBe(202);
    const limited = await POST(post(tripo, authed));
    expect(limited.status).toBe(429);
    expect(limited.headers.get("retry-after")).toBeTruthy();
  });
  it("relays provider rate limiting with Retry-After", async () => {
    fetchMock.mockImplementation(async () => new Response("", { status: 429, headers: { "retry-after": "9" } }));
    const res = await POST(post(tripo, authed));
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
    fetchMock.mockResolvedValue(task("success", { progress: 100, output: { model_url: "https://tripo-data.rg1.data.tripo3d.com/signed.glb?sig=secret" } }));
    const res = await GET(get(), ctx("task-abc-123"));
    const text = await res.text();
    expect(JSON.parse(text).task).toMatchObject({ status: "completed", hasModel: true });
    expect(text).not.toContain("signed.glb");
  });
  it("reports that Tripo cannot cancel (501) without calling the network", async () => {
    const res = await DELETE(get(), ctx("task-abc-123"));
    expect(res.status).toBe(501);
    expect(await res.json()).toMatchObject({ code: "cancel-unsupported", retryable: false });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("GET /api/generate/[taskId]/model", () => {
  const req = () => new Request("http://localhost/x", { headers: { "x-sift-access-code": "letmein" } });
  const glb = () => { const b = new Uint8Array(40); b.set([0x67, 0x6c, 0x54, 0x46]); return b; };
  it("re-reads the task for a fresh URL and streams a valid GLB", async () => {
    fetchMock
      .mockResolvedValueOnce(task("success", { output: { model_url: "https://tripo-data.rg1.data.tripo3d.com/fresh.glb" } }))
      .mockResolvedValueOnce(new Response(glb()));
    const res = await GET_MODEL(req(), ctx("task-abc-123"));
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("model/gltf-binary");
    expect((await res.arrayBuffer()).byteLength).toBe(40);
    expect(fetchMock.mock.calls[1][0]).toBe("https://tripo-data.rg1.data.tripo3d.com/fresh.glb");
  });
  it("refuses unfinished tasks and off-allowlist URLs", async () => {
    fetchMock.mockResolvedValueOnce(task("running"));
    expect((await GET_MODEL(req(), ctx("task-abc-123"))).status).toBe(409);
    fetchMock.mockResolvedValueOnce(task("success", { output: { model_url: "https://evil.example/x.glb" } }));
    const res = await GET_MODEL(req(), ctx("task-abc-123"));
    expect(res.status).toBe(502);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

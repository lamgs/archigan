import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetSpendLimiterForTests } from "@/lib/providers/hosted-http";
import { getProvider } from "@/lib/providers/registry";
import { HOSTED_PROVIDER_IDS, ProviderError } from "@/lib/providers/types";
import { GET as PROVIDERS } from "../providers/route";
import { DELETE, GET } from "./[taskId]/route";
import { GET as GET_MODEL } from "./[taskId]/model/route";
import { POST } from "./route";

// Tripo is the only hosted provider (ADR-018). Adapter methods are spied, so no network is used.
const TRIPO_ENV = { TRIPO_ENABLED: "true", TRIPO_API_KEY: "k-tripo" };
const LEGACY = ["meshy", "hunyuan3d-rapid", "hunyuan3d-pro", "tencent-rapid", "tencent-pro"];
const setEnv = (env: Record<string, string>) => Object.entries(env).forEach(([k, v]) => vi.stubEnv(k, v));
const post = (provider: string, headers: Record<string, string> = {}, extra: object = {}) => new Request("http://localhost/api/generate", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify({ prompt: "A terraced tower", refinement: "", provider, confirmSpend: true, ...extra }) });
const taskReq = (provider: string | null, headers: Record<string, string> = {}, method = "GET", path = "") => new Request(`http://localhost/api/generate/task-abc-123${path}${provider ? `?provider=${provider}` : ""}`, { method, headers });
const ctx = { params: Promise.resolve({ taskId: "task-abc-123" }) };
const authed = { "x-sift-access-code": "letmein", "x-forwarded-for": "9.9.9.9" };
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  resetSpendLimiterForTests();
  fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
  vi.spyOn(getProvider("tripo"), "create").mockResolvedValue("task-abc-123");
});
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("hosted provider tripo", () => {
  it("is the only registered hosted provider", () => {
    expect([...HOSTED_PROVIDER_IDS]).toEqual(["tripo"]);
  });
  it("fails closed without its own flag, key and a shared access code — and never calls the adapter", async () => {
    expect((await POST(post("tripo", authed))).status).toBe(503); // nothing configured
    setEnv({ SIFT_ACCESS_CODE: "letmein" });
    expect((await POST(post("tripo", authed))).status).toBe(503); // code only
    setEnv(TRIPO_ENV);
    vi.stubEnv("TRIPO_ENABLED", "false");
    expect((await POST(post("tripo", authed))).status).toBe(503); // flag off
    vi.stubEnv("TRIPO_ENABLED", "true");
    vi.stubEnv("TRIPO_API_KEY", "");
    expect((await POST(post("tripo", authed))).status).toBe(503); // key missing
    vi.stubEnv("TRIPO_API_KEY", "k-tripo");
    vi.stubEnv("SIFT_ACCESS_CODE", "");
    expect((await POST(post("tripo", authed))).status).toBe(503); // code missing
    expect(getProvider("tripo").create).not.toHaveBeenCalled();
  });
  it("does not honour the removed MESHY_ACCESS_CODE / MESHY_DAILY_LIMIT fallbacks", async () => {
    setEnv({ ...TRIPO_ENV, MESHY_ACCESS_CODE: "letmein" });
    expect((await POST(post("tripo", authed))).status).toBe(503);
  });
  it("requires the access code and an explicit confirmation when configured", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    expect((await POST(post("tripo"))).status).toBe(401);
    expect((await POST(post("tripo", { "x-sift-access-code": "nope" }))).status).toBe(401);
    const unconfirmed = await POST(post("tripo", authed, { confirmSpend: undefined }));
    expect(unconfirmed.status).toBe(400);
    expect(await unconfirmed.json()).toMatchObject({ code: "confirmation-required" });
    expect((await POST(post("tripo", authed, { confirmSpend: "yes" }))).status).toBe(400);
    expect(getProvider("tripo").create).not.toHaveBeenCalled();
    const ok = await POST(post("tripo", authed));
    expect(ok.status).toBe(202);
    expect(await ok.json()).toMatchObject({ kind: "task", provider: "tripo", taskId: "task-abc-123", verified: false });
    expect(getProvider("tripo").create).toHaveBeenCalledTimes(1);
  });
  it("status, cancel and model routes are guarded and never leak the model URL; cancel is unsupported (501)", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    const p = getProvider("tripo");
    vi.spyOn(p, "status").mockResolvedValue({ providerTaskId: "task-abc-123", status: "running", progress: 40, glbUrl: "https://signed.example/secret" });
    expect((await GET(taskReq("tripo"), ctx)).status).toBe(401);
    expect((await DELETE(taskReq("tripo", {}, "DELETE"), ctx)).status).toBe(401);
    expect((await GET_MODEL(taskReq("tripo", {}, "GET", "/model"), ctx)).status).toBe(401);
    const status = await GET(taskReq("tripo", authed), ctx);
    const text = await status.text();
    expect(JSON.parse(text).task).toMatchObject({ status: "running", hasModel: true });
    expect(text).not.toContain("signed.example");
    const cancel = await DELETE(taskReq("tripo", authed, "DELETE"), ctx);
    expect(cancel.status).toBe(501);
    expect(await cancel.json()).toMatchObject({ code: "cancel-unsupported" });
    vi.spyOn(p, "status").mockResolvedValue({ providerTaskId: "task-abc-123", status: "running" });
    expect((await GET_MODEL(taskReq("tripo", authed, "GET", "/model"), ctx)).status).toBe(409);
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("task routes are fail-closed when Tripo is unconfigured", async () => {
    setEnv({ SIFT_ACCESS_CODE: "letmein" });
    expect((await GET(taskReq("tripo", authed), ctx)).status).toBe(503);
    expect((await DELETE(taskReq("tripo", authed, "DELETE"), ctx)).status).toBe(503);
    expect((await GET_MODEL(taskReq("tripo", authed, "GET", "/model"), ctx)).status).toBe(503);
  });
});

describe("unknown and legacy providers", () => {
  it.each(LEGACY)("POST /api/generate rejects legacy provider %s with 400 unsupported-provider and never spends", async (id) => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    const res = await POST(post(id, authed));
    expect(res.status).toBe(400);
    expect(await res.json()).toMatchObject({ code: "unsupported-provider" });
    expect(getProvider("tripo").create).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });
  it("rejects unsupported providers before configuration or access-code checks (no 503/401 leak)", async () => {
    expect((await POST(post("meshy"))).status).toBe(400);
  });
  it("still answers 400 (invalid request) for ids that never existed", async () => {
    expect((await POST(post("rodin", authed))).status).toBe(400);
  });
  it.each([...LEGACY, "rodin"])("task routes answer 400 unknown-provider for ?provider=%s", async (id) => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    for (const res of [await GET(taskReq(id, authed), ctx), await DELETE(taskReq(id, authed, "DELETE"), ctx), await GET_MODEL(taskReq(id, authed, "GET", "/model"), ctx)]) {
      expect(res.status).toBe(400);
      expect(await res.json()).toMatchObject({ code: "unknown-provider" });
    }
  });
  it("a missing ?provider defaults to tripo", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    vi.spyOn(getProvider("tripo"), "status").mockResolvedValue({ providerTaskId: "task-abc-123", status: "queued" });
    expect((await GET(taskReq(null, authed), ctx)).status).toBe(200);
    expect(getProvider("tripo").status).toHaveBeenCalledTimes(1);
  });
});

describe("shared spend limits", () => {
  it("rate-limits per IP and by daily cap, with Retry-After", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    for (let i = 0; i < 3; i++) expect((await POST(post("tripo", authed))).status).toBe(202);
    const limited = await POST(post("tripo", authed));
    expect(limited.status).toBe(429);
    expect(limited.headers.get("retry-after")).toBeTruthy();
    expect(await limited.json()).toMatchObject({ code: "rate-limited" });
    vi.stubEnv("SIFT_DAILY_LIMIT", "2");
    resetSpendLimiterForTests();
    expect((await POST(post("tripo", { ...authed, "x-forwarded-for": "1.1.1.1" }))).status).toBe(202);
    expect((await POST(post("tripo", { ...authed, "x-forwarded-for": "2.2.2.2" }))).status).toBe(202);
    expect(await (await POST(post("tripo", { ...authed, "x-forwarded-for": "3.3.3.3" }))).json()).toMatchObject({ code: "daily-limit" });
  });
  it("ignores the removed MESHY_DAILY_LIMIT fallback", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein", MESHY_DAILY_LIMIT: "1" });
    for (const ip of ["1.1.1.1", "2.2.2.2"]) expect((await POST(post("tripo", { ...authed, "x-forwarded-for": ip }))).status).toBe(202);
  });
  it("failed provider calls do not consume the shared budget", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein" });
    vi.spyOn(getProvider("tripo"), "create").mockRejectedValue(new ProviderError("insufficient-credits", "no credits", 402, false));
    for (let i = 0; i < 5; i++) expect((await POST(post("tripo", authed))).status).toBe(402);
    vi.spyOn(getProvider("tripo"), "create").mockResolvedValue("task-abc-123");
    expect((await POST(post("tripo", authed))).status).toBe(202);
  });
});

describe("GET /api/providers", () => {
  it("lists only procedural and tripo, fail-closed, unverified, with a cost label and no secrets", async () => {
    setEnv({ ...TRIPO_ENV, SIFT_ACCESS_CODE: "letmein", MESHY_API_KEY: "k-meshy", FAL_KEY: "k-fal", TENCENT_SECRET_KEY: "k-tencent" });
    const text = await (await PROVIDERS()).text();
    const body = JSON.parse(text);
    expect(Object.keys(body).sort()).toEqual(["procedural", "tripo"]);
    expect(body.procedural).toMatchObject({ configured: true });
    expect(body.tripo).toMatchObject({ configured: true, verified: false, label: "Tripo", costLabel: "≈ $0.30 per model (estimate)", supportsCancel: false });
    ["k-tripo", "k-meshy", "k-fal", "k-tencent", "letmein"].forEach((secret) => expect(text).not.toContain(secret));
    vi.stubEnv("TRIPO_ENABLED", "false");
    expect((await (await PROVIDERS()).json()).tripo.configured).toBe(false);
  });
});

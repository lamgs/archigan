import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resetSpendLimiterForTests } from "@/lib/providers/hosted-http";
import { getProvider, listProviders } from "@/lib/providers/registry";
import { HOSTED_PROVIDER_IDS, type HostedProviderId } from "@/lib/providers/types";
import { GET as PROVIDERS } from "../providers/route";
import { DELETE, GET } from "./[taskId]/route";
import { GET as GET_MODEL } from "./[taskId]/model/route";
import { POST } from "./route";

// Provider selection + fail-closed behaviour across all hosted providers. Adapter methods are spied, so no network is used.
const ENV: Record<HostedProviderId, Record<string, string>> = {
  meshy: { MESHY_ENABLED: "true", MESHY_API_KEY: "k-meshy" },
  tripo: { TRIPO_ENABLED: "true", TRIPO_API_KEY: "k-tripo" },
  "hunyuan3d-rapid": { HUNYUAN_ENABLED: "true", FAL_KEY: "k-fal" },
  "hunyuan3d-pro": { HUNYUAN_ENABLED: "true", FAL_KEY: "k-fal" },
};
const setEnv = (env: Record<string, string>) => Object.entries(env).forEach(([k, v]) => vi.stubEnv(k, v));
const post = (provider: string, headers: Record<string, string> = {}, extra: object = {}) => new Request("http://localhost/api/generate", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify({ prompt: "A terraced tower", refinement: "", provider, confirmSpend: true, ...extra }) });
const taskReq = (provider: string | null, headers: Record<string, string> = {}, method = "GET", path = "") => new Request(`http://localhost/api/generate/task-abc-123${path}${provider ? `?provider=${provider}` : ""}`, { method, headers });
const ctx = { params: Promise.resolve({ taskId: "task-abc-123" }) };
const authed = { "x-sift-access-code": "letmein", "x-forwarded-for": "9.9.9.9" };
let fetchMock: ReturnType<typeof vi.fn>;

beforeEach(() => {
  resetSpendLimiterForTests();
  fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock);
  listProviders().forEach((p) => vi.spyOn(p, "create").mockResolvedValue("task-abc-123"));
});
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe.each(HOSTED_PROVIDER_IDS)("hosted provider %s", (id) => {
  it("fails closed without its own flag, key and a shared access code — and never calls the adapter", async () => {
    expect((await POST(post(id, authed))).status).toBe(503); // nothing configured
    setEnv({ SIFT_ACCESS_CODE: "letmein" });
    expect((await POST(post(id, authed))).status).toBe(503); // code only
    setEnv(ENV[id]);
    vi.stubEnv(Object.keys(ENV[id])[0], "false");
    expect((await POST(post(id, authed))).status).toBe(503); // flag off
    vi.stubEnv(Object.keys(ENV[id])[0], "true");
    vi.stubEnv(Object.keys(ENV[id])[1], "");
    expect((await POST(post(id, authed))).status).toBe(503); // key missing
    expect(getProvider(id).create).not.toHaveBeenCalled();
  });
  it("requires the access code and an explicit confirmation when configured", async () => {
    setEnv({ ...ENV[id], SIFT_ACCESS_CODE: "letmein" });
    expect((await POST(post(id))).status).toBe(401);
    expect((await POST(post(id, { "x-sift-access-code": "nope" }))).status).toBe(401);
    expect((await POST(post(id, authed, { confirmSpend: undefined }))).status).toBe(400);
    expect(getProvider(id).create).not.toHaveBeenCalled();
    const ok = await POST(post(id, authed));
    expect(ok.status).toBe(202);
    expect(await ok.json()).toMatchObject({ kind: "task", provider: id, taskId: "task-abc-123", verified: false });
    expect(getProvider(id).create).toHaveBeenCalledTimes(1);
  });
  it("accepts the legacy MESHY_ACCESS_CODE as a fallback shared code", async () => {
    setEnv({ ...ENV[id], MESHY_ACCESS_CODE: "letmein" });
    expect((await POST(post(id, authed))).status).toBe(202);
  });
  it("status, cancel and model routes use the selected provider and are guarded", async () => {
    setEnv({ ...ENV[id], SIFT_ACCESS_CODE: "letmein" });
    const p = getProvider(id);
    vi.spyOn(p, "status").mockResolvedValue({ providerTaskId: "task-abc-123", status: "running", progress: 40, glbUrl: "https://signed.example/secret" });
    vi.spyOn(p, "cancel").mockResolvedValue();
    expect((await GET(taskReq(id), ctx)).status).toBe(401);
    const status = await GET(taskReq(id, authed), ctx);
    const text = await status.text();
    expect(JSON.parse(text).task).toMatchObject({ status: "running", hasModel: true });
    expect(text).not.toContain("signed.example"); // model URL never reaches the browser
    expect((await DELETE(taskReq(id, authed, "DELETE"), ctx)).status).toBe(200);
    expect(p.cancel).toHaveBeenCalledTimes(1);
    vi.spyOn(p, "status").mockResolvedValue({ providerTaskId: "task-abc-123", status: "running" });
    expect((await GET_MODEL(taskReq(id, authed, "GET", "/model"), ctx)).status).toBe(409);
  });
});

describe("provider selection", () => {
  it("one configured provider does not enable another", async () => {
    setEnv({ ...ENV.meshy, SIFT_ACCESS_CODE: "letmein" });
    expect((await POST(post("meshy", authed))).status).toBe(202);
    expect((await POST(post("tripo", authed))).status).toBe(503);
    expect((await POST(post("hunyuan3d-pro", authed))).status).toBe(503);
    expect((await GET(taskReq("tripo", authed), ctx)).status).toBe(503);
  });
  it("rejects unknown providers on task routes and defaults a missing ?provider to meshy", async () => {
    setEnv({ ...ENV.meshy, SIFT_ACCESS_CODE: "letmein" });
    expect((await GET(taskReq("rodin", authed), ctx)).status).toBe(400);
    vi.spyOn(getProvider("meshy"), "status").mockResolvedValue({ providerTaskId: "task-abc-123", status: "queued" });
    expect((await GET(taskReq(null, authed), ctx)).status).toBe(200);
    expect((await POST(post("rodin", authed))).status).toBe(400);
  });
  it("counts per-IP and daily limits across all providers", async () => {
    Object.values(ENV).forEach(setEnv);
    setEnv({ SIFT_ACCESS_CODE: "letmein" });
    for (const id of ["meshy", "tripo", "hunyuan3d-rapid"]) expect((await POST(post(id, authed))).status).toBe(202);
    const limited = await POST(post("hunyuan3d-pro", authed));
    expect(limited.status).toBe(429);
    expect(await limited.json()).toMatchObject({ code: "rate-limited" });
    vi.stubEnv("SIFT_DAILY_LIMIT", "2");
    resetSpendLimiterForTests();
    expect((await POST(post("tripo", { ...authed, "x-forwarded-for": "1.1.1.1" }))).status).toBe(202);
    expect((await POST(post("meshy", { ...authed, "x-forwarded-for": "2.2.2.2" }))).status).toBe(202);
    expect(await (await POST(post("hunyuan3d-rapid", { ...authed, "x-forwarded-for": "3.3.3.3" }))).json()).toMatchObject({ code: "daily-limit" });
  });
  it("failed provider calls do not consume the shared budget", async () => {
    setEnv({ ...ENV.tripo, SIFT_ACCESS_CODE: "letmein" });
    const { ProviderError } = await import("@/lib/providers/types");
    vi.spyOn(getProvider("tripo"), "create").mockRejectedValue(new ProviderError("insufficient-credits", "no credits", 402, false));
    for (let i = 0; i < 5; i++) expect((await POST(post("tripo", authed))).status).toBe(402);
    vi.spyOn(getProvider("tripo"), "create").mockResolvedValue("task-abc-123");
    expect((await POST(post("tripo", authed))).status).toBe(202);
  });
});

describe("GET /api/providers", () => {
  it("lists every provider, fail-closed, unverified, with cost labels and no secrets", async () => {
    Object.values(ENV).forEach(setEnv);
    setEnv({ SIFT_ACCESS_CODE: "letmein" });
    vi.stubEnv("TRIPO_ENABLED", "false");
    const text = await (await PROVIDERS()).text();
    const body = JSON.parse(text);
    expect(body.procedural).toMatchObject({ configured: true });
    for (const id of HOSTED_PROVIDER_IDS) expect(body[id]).toMatchObject({ verified: false, label: expect.any(String), costLabel: expect.any(String) });
    expect(body.meshy.configured).toBe(true);
    expect(body.tripo.configured).toBe(false);
    ["k-meshy", "k-tripo", "k-fal", "letmein"].forEach((secret) => expect(text).not.toContain(secret));
  });
});

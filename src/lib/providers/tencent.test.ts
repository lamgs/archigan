import { createHash, createHmac } from "node:crypto";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { downloadGlbFor } from "./http";
import { mapTencentError, signTc3, tencentProProvider, tencentRapidProvider } from "./tencent";
import type { HostedProvider } from "./types";

// Documented-contract tests against a mocked fetch. The Tencent contract is UNVERIFIED; nothing here proves live behaviour.
const SID = "AKIDtestsecretid0000";
const SKEY = "testsecretkey0000";
const json = (body: unknown, init: ResponseInit = {}) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" }, ...init });
const wrap = (r: Record<string, unknown>) => json({ Response: { RequestId: "req-1", ...r } });
const glb = (extra = 20) => { const b = new Uint8Array(12 + extra); b.set([0x67, 0x6c, 0x54, 0x46]); return b.buffer; };
const mock = (impl: (url: string, init: RequestInit) => Response | Promise<Response>) => vi.fn(async (url: unknown, init: unknown) => impl(String(url), init as RequestInit)) as unknown as typeof fetch & ReturnType<typeof vi.fn>;
const calls = (f: unknown) => (f as ReturnType<typeof vi.fn>).mock.calls as [string, RequestInit][];
const hdr = (f: unknown, i = 0) => calls(f)[i][1].headers as Record<string, string>;
const ID = "job-0a1b2c3d4e5f";

const tiers: [string, HostedProvider, string, string, number][] = [
  ["rapid", tencentRapidProvider, "SubmitHunyuanTo3DRapidJob", "QueryHunyuanTo3DRapidJob", 200],
  ["pro", tencentProProvider, "SubmitHunyuanTo3DProJob", "QueryHunyuanTo3DProJob", 1024],
];

beforeEach(() => { vi.stubEnv("TENCENT_HY3D_ENABLED", "true"); vi.stubEnv("TENCENT_SECRET_ID", SID); vi.stubEnv("TENCENT_SECRET_KEY", SKEY); vi.stubEnv("SIFT_ACCESS_CODE", "letmein"); });
afterEach(() => { vi.unstubAllEnvs(); vi.useRealTimers(); });

describe("signTc3", () => {
  // No official known-answer vector was available (Tencent's docs mask the secret key), so this re-derives the
  // signature step by step with node:crypto, independently of the implementation.
  it("matches an independent step-by-step derivation", () => {
    const ts = 1551113065; // 2019-02-25 UTC
    const payload = '{"Prompt":"a tower"}';
    const sha = (s: string) => createHash("sha256").update(s).digest("hex");
    const h = (k: string | Buffer, d: string) => createHmac("sha256", k).update(d).digest();
    const canonical = `POST\n/\n\ncontent-type:application/json; charset=utf-8\nhost:ai3d.intl.tencentcloudapi.com\nx-tc-action:submithunyuanto3drapidjob\n\ncontent-type;host;x-tc-action\n${sha(payload)}`;
    const sts = `TC3-HMAC-SHA256\n${ts}\n2019-02-25/ai3d/tc3_request\n${sha(canonical)}`;
    const expected = h(h(h(h("TC3" + SKEY, "2019-02-25"), "ai3d"), "tc3_request"), sts).toString("hex");
    const out = signTc3({ secretId: SID, secretKey: SKEY, service: "ai3d", host: "ai3d.intl.tencentcloudapi.com", action: "SubmitHunyuanTo3DRapidJob", version: "2025-05-13", region: "ap-guangzhou", payload, timestamp: ts });
    expect(out.canonicalRequest).toBe(canonical);
    expect(out.signature).toBe(expected);
    expect(out.authorization).toBe(`TC3-HMAC-SHA256 Credential=${SID}/2019-02-25/ai3d/tc3_request, SignedHeaders=content-type;host;x-tc-action, Signature=${expected}`);
    expect(out.headers["X-TC-Timestamp"]).toBe("1551113065");
  });
  it("changes with payload, timestamp date and key", () => {
    const base = { secretId: SID, secretKey: SKEY, service: "ai3d", host: "h", action: "A", version: "v", payload: "{}", timestamp: 1_700_000_000 };
    const s = signTc3(base).signature;
    expect(signTc3({ ...base, payload: "{ }" }).signature).not.toBe(s);
    expect(signTc3({ ...base, timestamp: 1_700_000_000 + 86_400 }).signature).not.toBe(s);
    expect(signTc3({ ...base, secretKey: "other" }).signature).not.toBe(s);
  });
});

describe("metadata", () => {
  it("exposes ids, labels, costs, no cancel, COS hosts", () => {
    expect(tencentRapidProvider).toMatchObject({ id: "tencent-rapid", label: "HY 3D Rapid (Tencent)", supportsCancel: false });
    expect(tencentProProvider).toMatchObject({ id: "tencent-pro", label: "HY 3D Pro (Tencent)", supportsCancel: false });
    expect(tencentRapidProvider.costLabel).toMatch(/15 credits.*estimate/);
    expect(tencentProProvider.costLabel).toMatch(/25 credits.*estimate/);
    expect(tencentRapidProvider.assetHosts).toContain("myqcloud.com");
  });
});

describe.each(tiers)("%s create", (name, provider, submit, _q, limit) => {
  it("POSTs a signed request and returns JobId", async () => {
    const f = mock(() => wrap({ JobId: ID }));
    await expect(provider.create("A tower", "with terraces", f)).resolves.toBe(ID);
    const [url, init] = calls(f)[0];
    expect(url).toBe("https://ai3d.intl.tencentcloudapi.com/");
    expect(init.method).toBe("POST");
    const h = hdr(f);
    expect(h["X-TC-Action"]).toBe(submit);
    expect(h["X-TC-Version"]).toBe("2025-05-13");
    expect(h["X-TC-Region"]).toBeTruthy();
    expect(h["Content-Type"]).toMatch(/^application\/json/);
    expect(h["X-TC-Timestamp"]).toMatch(/^\d{10}$/);
    expect(h.Authorization).toMatch(new RegExp(`^TC3-HMAC-SHA256 Credential=${SID}/\\d{4}-\\d{2}-\\d{2}/ai3d/tc3_request, SignedHeaders=content-type;host;x-tc-action, Signature=[0-9a-f]{64}$`));
    const body = JSON.parse(String(init.body));
    expect(body.Prompt).toMatch(/a tower with terraces/);
    expect(new TextEncoder().encode(body.Prompt).length).toBeLessThanOrEqual(limit);
    if (name === "rapid") expect(body.ResultFormat).toBe("GLB");
    else { expect(body.ResultFormat).toBeUndefined(); expect(body.FaceCount).toBeGreaterThanOrEqual(3000); expect(body.FaceCount).toBeLessThanOrEqual(1_500_000); }
  });
  it("signs the exact body it sends", async () => {
    vi.useFakeTimers(); vi.setSystemTime(new Date("2025-06-01T00:00:00Z"));
    const f = mock(() => wrap({ JobId: ID }));
    await provider.create("A tower", "", f);
    const init = calls(f)[0][1];
    const expected = signTc3({ secretId: SID, secretKey: SKEY, service: "ai3d", host: "ai3d.intl.tencentcloudapi.com", action: submit, version: "2025-05-13", region: hdr(f)["X-TC-Region"], payload: String(init.body), timestamp: Date.parse("2025-06-01T00:00:00Z") / 1000 });
    expect(hdr(f).Authorization).toBe(expected.authorization);
  });
  it("keeps long and multibyte prompts within the limit", async () => {
    for (const brief of ["x".repeat(3000), "塔".repeat(3000), "tower 🏗 ".repeat(300)]) {
      const f = mock(() => wrap({ JobId: ID }));
      await provider.create(brief, "", f);
      const prompt = JSON.parse(String(calls(f)[0][1].body)).Prompt as string;
      expect(new TextEncoder().encode(prompt).length).toBeLessThanOrEqual(limit);
      expect(prompt.length).toBeGreaterThan(0);
      expect(prompt).not.toContain("�");
    }
  });
  it("rejects missing or malformed job ids with bad-response", async () => {
    for (const r of [{}, { JobId: "short" }, { JobId: "bad id!!" }, { JobId: 12345678 }, { JobId: "../../etc/passwd" }]) {
      await expect(provider.create("a tower", "", mock(() => wrap(r)))).rejects.toMatchObject({ code: "bad-response" });
    }
  });
  it("refuses without credentials and makes no call", async () => {
    for (const v of ["TENCENT_SECRET_ID", "TENCENT_SECRET_KEY"]) {
      vi.stubEnv(v, "");
      const f = mock(() => wrap({ JobId: ID }));
      await expect(provider.create("a tower", "", f)).rejects.toMatchObject({ code: "not-configured" });
      expect(calls(f)).toHaveLength(0);
      vi.stubEnv(v, v === "TENCENT_SECRET_ID" ? SID : SKEY);
    }
  });
});

describe.each(tiers)("%s status", (_n, provider, _s, query) => {
  const done = (files: unknown) => mock(() => wrap({ Status: "DONE", ResultFile3Ds: files }));
  it.each([["WAIT", "queued"], ["RUN", "running"]] as const)("maps %s -> %s", async (raw, status) => {
    const f = mock(() => wrap({ Status: raw }));
    await expect(provider.status(ID, f)).resolves.toEqual({ providerTaskId: ID, status });
    expect(hdr(f)["X-TC-Action"]).toBe(query);
    expect(JSON.parse(String(calls(f)[0][1].body))).toEqual({ JobId: ID });
  });
  it("DONE returns the GLB url, preferring GLB over OBJ", async () => {
    const task = await provider.status(ID, done([{ Type: "OBJ", Url: "https://b.cos.ap-guangzhou.myqcloud.com/m.obj" }, { Type: "GLB", Url: "https://b.cos.ap-guangzhou.myqcloud.com/m.glb?sign=x" }]));
    expect(task).toMatchObject({ status: "completed", glbUrl: "https://b.cos.ap-guangzhou.myqcloud.com/m.glb?sign=x" });
    expect(task.expiresAt).toBeTruthy();
  });
  it("accepts an untyped entry ending in .glb", async () => {
    await expect(provider.status(ID, done([{ Url: "https://b.myqcloud.com/m.glb" }]))).resolves.toMatchObject({ status: "completed" });
  });
  it("OBJ/FBX-only results fail with not-glb; empty with no-glb", async () => {
    await expect(provider.status(ID, done([{ Type: "OBJ", Url: "https://b.myqcloud.com/m.obj" }, { Type: "FBX", Url: "https://b.myqcloud.com/m.fbx" }]))).resolves.toMatchObject({ status: "failed", error: { code: "not-glb" } });
    await expect(provider.status(ID, done([]))).resolves.toMatchObject({ status: "failed", error: { code: "no-glb" } });
    await expect(provider.status(ID, done(undefined))).resolves.toMatchObject({ status: "failed", error: { code: "no-glb" } });
  });
  it("FAIL becomes a failed task without echoing provider text", async () => {
    const t = await provider.status(ID, mock(() => wrap({ Status: "FAIL", ErrorCode: "X", ErrorMessage: "secret internal detail" })));
    expect(t).toMatchObject({ status: "failed", error: { code: "provider-failed" } });
    expect(JSON.stringify(t)).not.toContain("secret internal detail");
  });
  it("treats unknown status/garbage as failure or bad-response", async () => {
    await expect(provider.status(ID, mock(() => wrap({ Status: "WEIRD" })))).resolves.toMatchObject({ status: "failed", error: { code: "unknown-status" } });
    await expect(provider.status(ID, mock(() => json("nope")))).rejects.toMatchObject({ code: "bad-response" });
  });
  it("rejects ids that are malformed and never fetches", async () => {
    const f = mock(() => wrap({}));
    for (const id of ["../../x", "a b c d e f", "short", "x".repeat(81)]) await expect(provider.status(id, f)).rejects.toMatchObject({ code: "bad-task-id" });
    expect(calls(f)).toHaveLength(0);
  });
});

describe("errors", () => {
  const p = tencentRapidProvider;
  it.each([
    ["AuthFailure.SignatureFailure", "auth", 401, false], ["AuthFailure.SecretIdNotFound", "auth", 401, false],
    ["RequestLimitExceeded", "rate-limited", 429, true], ["LimitExceeded.JobNumLimit", "rate-limited", 429, true],
    ["ResourceInsufficient", "insufficient-credits", 402, false], ["FailedOperation.InsufficientBalance", "insufficient-credits", 402, false], ["OperationDenied.AccountArrears", "insufficient-credits", 402, false],
    ["InvalidParameterValue.Prompt", "rejected", 400, false], ["InvalidParameter", "rejected", 400, false],
    ["InternalError", "provider-unavailable", 503, true], ["ServiceUnavailable", "provider-unavailable", 503, true],
  ] as const)("Tencent %s -> %s", async (Code, code, httpStatus, retryable) => {
    await expect(p.create("a tower", "", mock(() => wrap({ Error: { Code, Message: "m" } })))).rejects.toMatchObject({ code, httpStatus, retryable });
    await expect(p.status(ID, mock(() => wrap({ Error: { Code, Message: "m" } })))).rejects.toMatchObject({ code });
  });
  it("unknown error codes are rejected generically", () => expect(mapTencentError("Weird.Thing")).toMatchObject({ code: "rejected", retryable: false }));
  it.each([[401, "auth"], [403, "auth"], [404, "not-found"], [429, "rate-limited"], [400, "rejected"], [500, "provider-unavailable"], [503, "provider-unavailable"]] as const)("HTTP %i -> %s", async (status, code) => {
    await expect(p.create("a tower", "", mock(() => new Response("<html/>", { status })))).rejects.toMatchObject({ code });
    await expect(p.status(ID, mock(() => new Response("{}", { status })))).rejects.toMatchObject({ code });
  });
  it("honours Retry-After on HTTP 429", async () => {
    await expect(p.create("a tower", "", mock(() => new Response("{}", { status: 429, headers: { "retry-after": "30" } })))).rejects.toMatchObject({ retryAfterSeconds: 30, retryable: true });
  });
  it("maps timeouts and network failures", async () => {
    await expect(p.create("a tower", "", mock(() => { throw new DOMException("t", "TimeoutError"); }))).rejects.toMatchObject({ code: "timeout", retryable: true });
    await expect(p.status(ID, mock(() => { throw new TypeError("fetch failed"); }))).rejects.toMatchObject({ code: "network" });
  });
  it("never leaks secrets, signatures, or response bodies in errors", async () => {
    const leaky = mock(() => json({ Response: { Error: { Code: "AuthFailure.SignatureFailure", Message: `bad ${SKEY} ${SID}` } } }));
    const throwing = mock(() => { throw new Error(`conn ${SKEY} ${SID}`); });
    const errs = await Promise.all([p.create("a tower", "", leaky), p.status(ID, leaky), p.create("a tower", "", throwing), p.status(ID, mock(() => new Response(`oops ${SKEY}`, { status: 500 })))].map((x) => x.catch((e) => e)));
    for (const e of errs) {
      expect(e).toBeInstanceOf(Error);
      const text = JSON.stringify({ m: (e as Error).message, ...(e as object) });
      expect(text).not.toContain(SKEY); expect(text).not.toContain(SID); expect(text).not.toMatch(/[0-9a-f]{64}/);
    }
  });
});

describe("cancel", () => {
  it.each([tencentRapidProvider, tencentProProvider])("$id is unsupported and makes no network call", async (p) => {
    const f = mock(() => wrap({}));
    await expect(p.cancel(ID, f)).rejects.toMatchObject({ code: "cancel-unsupported", httpStatus: 501, retryable: false });
    expect(calls(f)).toHaveLength(0);
  });
});

describe("config (fail closed)", () => {
  const base = { TENCENT_HY3D_ENABLED: "true", TENCENT_SECRET_ID: "i", TENCENT_SECRET_KEY: "k", SIFT_ACCESS_CODE: "c" };
  it.each([tencentRapidProvider, tencentProProvider])("$id requires flag, both keys, and an access code", (p) => {
    expect(p.config(base)).toMatchObject({ configured: true, verified: false });
    expect(p.config({ ...base, TENCENT_HY3D_ENABLED: "false" }).configured).toBe(false);
    expect(p.config({ ...base, TENCENT_HY3D_ENABLED: undefined }).configured).toBe(false);
    expect(p.config({ ...base, TENCENT_SECRET_ID: "" })).toMatchObject({ configured: false, hasKey: false });
    expect(p.config({ ...base, TENCENT_SECRET_KEY: undefined })).toMatchObject({ configured: false, hasKey: false });
    expect(p.config({ ...base, SIFT_ACCESS_CODE: undefined })).toMatchObject({ configured: false, accessCodeRequired: false });
    expect(p.config({ ...base, SIFT_ACCESS_CODE: undefined, MESHY_ACCESS_CODE: "m" }).configured).toBe(true);
    expect(p.config({ TENCENT_HY3D_ENABLED: "true", TENCENT_SECRET_ID: "i", TENCENT_SECRET_KEY: "k" }).configured).toBe(false);
  });
});

describe("downloadGlbFor", () => {
  const p = tencentRapidProvider;
  const ok = "https://bucket-123.cos.ap-guangzhou.myqcloud.com/out/model.glb?sign=abc";
  it("downloads a GLB from an allowlisted host", async () => {
    await expect(downloadGlbFor(p, ok, mock(() => new Response(glb())))).resolves.toBeInstanceOf(ArrayBuffer);
  });
  it("rejects foreign hosts, http, lookalikes without fetching", async () => {
    const f = mock(() => new Response(glb()));
    for (const url of ["https://evil.example/m.glb", "http://b.myqcloud.com/m.glb", "https://notmyqcloud.com/m.glb", "https://myqcloud.com.evil.com/m.glb"]) {
      await expect(downloadGlbFor(p, url, f)).rejects.toMatchObject({ code: "bad-asset-url" });
    }
    expect(calls(f)).toHaveLength(0);
  });
  it("enforces the size cap and the GLB magic number", async () => {
    await expect(downloadGlbFor({ ...p, maxGlbBytes: 16 }, ok, mock(() => new Response(glb(100))))).rejects.toMatchObject({ code: "too-large" });
    await expect(downloadGlbFor(p, ok, mock(() => new Response("<html>not a model</html>")))).rejects.toMatchObject({ code: "not-glb" });
  });
});

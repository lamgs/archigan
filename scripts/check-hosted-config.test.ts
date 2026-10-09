import { describe, expect, it } from "vitest";
// @ts-expect-error plain ESM script without types
import { evaluate } from "./check-hosted-config.mjs";

const CODE = "a-long-random-access-code";
describe("hosted config preflight", () => {
  it("is not ready without an access code, even with flag and key", () => {
    const r = evaluate({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "x" });
    expect(r.providers.find((p: { id: string }) => p.id === "tripo").configured).toBe(false);
    expect(r.warnings.join(" ")).toMatch(/No access code/);
  });
  it("requires flag AND key AND code, and only knows Tripo", () => {
    const by = (env: Record<string, string>) => Object.fromEntries(evaluate(env).providers.map((p: { id: string }) => [p.id, p]));
    expect(Object.keys(by({}))).toEqual(["tripo"]);
    expect(by({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "y", SIFT_ACCESS_CODE: CODE }).tripo.configured).toBe(true);
    const keyOnly = by({ TRIPO_API_KEY: "y", SIFT_ACCESS_CODE: CODE }).tripo;
    expect(keyOnly.configured).toBe(false);
    expect(keyOnly.problems.join(" ")).toMatch(/is not "true"/);
  });
  it("flags enabled-without-key, bad flag values, weak codes and NEXT_PUBLIC secrets", () => {
    const r = evaluate({ TRIPO_ENABLED: "yes", SIFT_ACCESS_CODE: "short", NEXT_PUBLIC_TRIPO_API_KEY: "x" });
    expect(JSON.stringify(r)).toMatch(/must be exactly/);
    expect(JSON.stringify(r)).toMatch(/shorter than 12/);
    expect(JSON.stringify(r)).toMatch(/NEXT_PUBLIC_TRIPO_API_KEY is NEXT_PUBLIC_/);
    expect(JSON.stringify(evaluate({ TRIPO_ENABLED: "true", SIFT_ACCESS_CODE: CODE }))).toMatch(/Enabled but missing TRIPO_API_KEY/);
  });
  it("does not honour removed-provider settings or the MESHY_ACCESS_CODE fallback, and says so", () => {
    const r = evaluate({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "y", MESHY_ACCESS_CODE: "a-long-random-access-code", MESHY_ENABLED: "true", FAL_KEY: "z" });
    expect(r.providers[0].configured).toBe(false);
    expect(r.accessCodeSet).toBe(false);
    expect(r.warnings.join(" ")).toMatch(/removed providers.*MESHY_ACCESS_CODE.*MESHY_ENABLED.*FAL_KEY/);
  });
  it("never includes secret values in its output", () => {
    expect(JSON.stringify(evaluate({ TRIPO_API_KEY: "SECRET-VALUE-123", SIFT_ACCESS_CODE: "ACCESS-VALUE-4567890" }))).not.toMatch(/SECRET-VALUE|ACCESS-VALUE/);
  });
  it("treats REPLACE_ME placeholders as missing", () => {
    const r = evaluate({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "REPLACE_ME", SIFT_ACCESS_CODE: CODE });
    expect(r.providers.find((p: { id: string }) => p.id === "tripo").configured).toBe(false);
    expect(JSON.stringify(r)).toMatch(/Enabled but missing TRIPO_API_KEY/);
  });
});

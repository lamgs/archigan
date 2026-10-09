import { describe, expect, it } from "vitest";
// @ts-expect-error plain ESM script without types
import { evaluate } from "./check-hosted-config.mjs";

const CODE = "a-long-random-access-code";
describe("hosted config preflight", () => {
  it("is not ready without an access code, even with flag and key", () => {
    const r = evaluate({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "x" });
    expect(r.providers.find((p: { label: string }) => p.label === "Tripo").configured).toBe(false);
    expect(r.warnings.join(" ")).toMatch(/No access code/);
  });
  it("requires flag AND key AND code; one provider does not enable another", () => {
    const r = evaluate({ HUNYUAN_ENABLED: "true", FAL_KEY: "x", SIFT_ACCESS_CODE: CODE, TRIPO_API_KEY: "y" });
    const by = Object.fromEntries(r.providers.map((p: { label: string; configured: boolean; problems: string[] }) => [p.label, p]));
    expect(by["Hunyuan3D via fal.ai"].configured).toBe(true);
    expect(by.Tripo.configured).toBe(false);
    expect(by.Tripo.problems.join(" ")).toMatch(/is not "true"/);
  });
  it("flags enabled-without-key, bad flag values, weak codes, the legacy code and NEXT_PUBLIC secrets", () => {
    const r = evaluate({ MESHY_ENABLED: "yes", TRIPO_ENABLED: "true", MESHY_ACCESS_CODE: "short", NEXT_PUBLIC_TRIPO_API_KEY: "x" });
    const text = JSON.stringify(r);
    expect(text).toMatch(/Enabled but missing TRIPO_API_KEY/);
    expect(text).toMatch(/must be exactly/);
    expect(text).toMatch(/shorter than 12/);
    expect(text).toMatch(/legacy MESHY_ACCESS_CODE/);
    expect(text).toMatch(/NEXT_PUBLIC_TRIPO_API_KEY is NEXT_PUBLIC_/);
  });
  it("never includes secret values in its output", () => {
    expect(JSON.stringify(evaluate({ FAL_KEY: "SECRET-VALUE-123", SIFT_ACCESS_CODE: "ACCESS-VALUE-4567890" }))).not.toMatch(/SECRET-VALUE|ACCESS-VALUE/);
  });
  it("treats REPLACE_ME placeholders as missing", () => {
    const r = evaluate({ TRIPO_ENABLED: "true", TRIPO_API_KEY: "REPLACE_ME", SIFT_ACCESS_CODE: CODE });
    expect(r.providers.find((p: { label: string }) => p.label === "Tripo").configured).toBe(false);
    expect(JSON.stringify(r)).toMatch(/Enabled but missing TRIPO_API_KEY/);
  });
});

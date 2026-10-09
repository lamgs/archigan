import { describe, expect, it } from "vitest";
import { PICKER_ORDER, PROVIDER_META, isHostedProvider, isSupportedProvider, providerCost, providerLabel, providerSupportsCancel } from "./provider-meta";

describe("provider metadata (ADR-018)", () => {
  it("offers only Local and Tripo", () => {
    expect(PICKER_ORDER).toEqual(["procedural", "tripo"]);
    expect(Object.keys(PROVIDER_META).sort()).toEqual(["procedural", "tripo"]);
    expect(providerCost("procedural")).toBe("Free");
    expect(providerCost("tripo")).toBe("≈ $0.30 per model (estimate)");
    expect(providerLabel("tripo")).toBe("Tripo");
    expect(providerSupportsCancel("tripo")).toBe(false);
    expect(isHostedProvider("tripo")).toBe(true);
    expect(isHostedProvider("procedural")).toBe(false);
  });
  it("falls back to the raw id for legacy or unknown providers without throwing", () => {
    for (const id of ["meshy", "hunyuan3d-pro", "tencent-rapid", "rodin"]) {
      expect(providerLabel(id)).toBe(id);
      expect(providerCost(id)).toBe("unknown");
      expect(providerSupportsCancel(id)).toBe(false);
      expect(isSupportedProvider(id)).toBe(false);
      expect(isHostedProvider(id)).toBe(false);
    }
  });
  it("prefers the server catalog when present", () => {
    expect(providerLabel("tripo", { tripo: { label: "Tripo X", costLabel: "$1" } })).toBe("Tripo X");
    expect(providerCost("tripo", { tripo: { costLabel: "$1" } })).toBe("$1");
  });
});

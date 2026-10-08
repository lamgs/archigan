import { describe, expect, it } from "vitest";
import { generateRequestSchema, siftProjectSchema } from "./contracts";
import { legacySamples as sampleProjects } from "./legacy-fixtures";

describe("contracts", () => {
  it("accepts every bundled sample", () => {
    sampleProjects.forEach((project) => expect(siftProjectSchema.parse(project)).toEqual(project));
  });

  it("rejects unknown providers", () => {
    expect(generateRequestSchema.safeParse({ prompt: "A library", refinement: "", provider: "browser-key" }).success).toBe(false);
  });
});

describe("provider enum backward compatibility (ADR-017)", () => {
  it("still accepts the original values and every newer hosted provider for settings and jobs", async () => {
    const { projectSettingsSchema, providerSchema } = await import("./contracts");
    for (const provider of ["procedural", "meshy", "tripo", "hunyuan3d-rapid", "hunyuan3d-pro"]) {
      expect(providerSchema.safeParse(provider).success).toBe(true);
      expect(projectSettingsSchema.safeParse({ provider }).success).toBe(true);
    }
    expect(providerSchema.safeParse("rodin").success).toBe(false);
  });
  it("opens a stored project saved with `meshy` or `procedural`", () => {
    for (const provider of ["procedural", "meshy"]) expect(generateRequestSchema.safeParse({ prompt: "A library", provider, confirmSpend: provider === "meshy" }).success).toBe(true);
  });
});

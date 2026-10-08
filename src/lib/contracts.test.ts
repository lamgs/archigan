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

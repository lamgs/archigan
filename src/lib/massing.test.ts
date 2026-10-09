import { describe, expect, it } from "vitest";
import { composeArchitecturalPrompt, normalizeBrief } from "./massing";

describe("massing domain", () => {
  it("normalizes whitespace and case", () => {
    expect(normalizeBrief("  Quiet   MUSEUM ", " Courtyard  ")).toBe("quiet museum courtyard");
  });

  it("adds safety framing to hosted prompts", () => {
    expect(composeArchitecturalPrompt("library")).toContain("no people");
  });
});


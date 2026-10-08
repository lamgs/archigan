import { describe, expect, it } from "vitest";
import { composeArchitecturalPrompt, deriveMassing, normalizeBrief } from "./massing";

describe("massing domain", () => {
  it("normalizes whitespace and case", () => {
    expect(normalizeBrief("  Quiet   MUSEUM ", " Courtyard  ")).toBe("quiet museum courtyard");
  });

  it("is deterministic for equivalent briefs", () => {
    expect(deriveMassing("Twisting tower", "terraced garden")).toEqual(
      deriveMassing("  TWISTING tower ", "terraced   garden"),
    );
  });

  it("maps architectural cues to explicit parameters", () => {
    const model = deriveMassing("A tall twisting glass tower around an atrium");
    expect(model.floors).toBeGreaterThanOrEqual(18);
    expect(model.twist).toBeGreaterThan(0);
    expect(model.courtyard).toBe(true);
    expect(model.material).toBe("glass");
  });

  it("adds safety framing to hosted prompts", () => {
    expect(composeArchitecturalPrompt("library")).toContain("no people");
  });
});


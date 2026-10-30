import { describe, expect, it } from "vitest";
import { normalizeBrief } from "./massing";

describe("massing domain", () => {
  it("normalizes whitespace and case", () => {
    expect(normalizeBrief("  Quiet   MUSEUM ", " Courtyard  ")).toBe("quiet museum courtyard");
  });
});


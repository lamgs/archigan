import { describe, expect, it } from "vitest";
import { composeArchitecturalPrompt, negativePromptFor, STYLE_CLAUSES } from "./architectural-style";

describe("composeArchitecturalPrompt", () => {
  it("leads with the brief in the user's own wording and adds the architectural style", () => {
    const text = composeArchitecturalPrompt("A Wooden Cabin", "by a lake");
    expect(text.startsWith("A Wooden Cabin by a lake.")).toBe(true);
    expect(text).toContain(STYLE_CLAUSES[0]);
    expect(text).toMatch(/\.$/);
  });
  it("drops lowest-priority clauses first when space is short, never the brief", () => {
    const brief = "x".repeat(700);
    const text = composeArchitecturalPrompt(brief, "", 1024);
    expect(text.length).toBeLessThanOrEqual(1024);
    expect(text.startsWith(brief)).toBe(true);
    expect(text).toContain(STYLE_CLAUSES[0]);
    expect(text).not.toContain(STYLE_CLAUSES[STYLE_CLAUSES.length - 1]);
  });
});

describe("negativePromptFor", () => {
  it("lists vernacular and toy-like looks by default", () => {
    expect(negativePromptFor("a wooden cabin")).toMatch(/vernacular, traditional, historic, gothic/);
  });
  it("removes any term whose words the brief itself uses", () => {
    const negative = negativePromptFor("a traditional village house with a steep roof");
    expect(negative).not.toMatch(/traditional|village house|steep gabled roof/);
    expect(negative).toMatch(/gothic/);
  });
});

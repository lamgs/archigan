import type { MassingSpec } from "./contracts";

export function normalizeBrief(prompt: string, refinement = "") {
  return `${prompt.trim().replace(/\s+/g, " ")} ${refinement.trim().replace(/\s+/g, " ")}`
    .trim()
    .toLowerCase();
}

function hashString(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function includesAny(text: string, terms: string[]) {
  return terms.some((term) => text.includes(term));
}

export function deriveMassing(prompt: string, refinement = ""): MassingSpec {
  const brief = normalizeBrief(prompt, refinement);
  const seed = hashString(brief);
  const tall = includesAny(brief, ["tower", "high-rise", "vertical", "skyscraper"]);
  const low = includesAny(brief, ["pavilion", "low-rise", "courtyard house", "single-story"]);
  const broad = includesAny(brief, ["campus", "museum", "cultural center", "horizontal"]);
  const slender = includesAny(brief, ["slender", "needle", "narrow"]);

  const material: MassingSpec["material"] = includesAny(brief, ["brick", "terracotta", "clay"])
    ? "terracotta"
    : includesAny(brief, ["glass", "transparent", "crystalline"])
      ? "glass"
      : includesAny(brief, ["concrete", "brutalist", "monolithic"])
        ? "concrete"
        : "limestone";

  return {
    seed,
    floors: low ? 3 + (seed % 3) : tall ? 18 + (seed % 15) : 7 + (seed % 8),
    width: slender ? 18 + (seed % 8) : broad ? 48 + (seed % 19) : 28 + (seed % 17),
    depth: broad ? 38 + ((seed >>> 4) % 16) : 24 + ((seed >>> 4) % 15),
    floorHeight: includesAny(brief, ["gallery", "museum", "atrium"]) ? 4.8 : 3.6,
    twist: includesAny(brief, ["twist", "spiral", "dynamic"]) ? 0.16 + (seed % 12) / 100 : 0,
    terrace: includesAny(brief, ["terrace", "stepped", "setback", "garden"]) ? 0.035 + (seed % 5) / 100 : 0.008,
    courtyard: includesAny(brief, ["courtyard", "atrium", "void", "hollow"]),
    material,
  };
}

export function composeArchitecturalPrompt(prompt: string, refinement = "") {
  const clean = normalizeBrief(prompt, refinement);
  return `Architectural concept massing model of ${clean}. Standalone building, clean geometry, coherent structure, no people, no vehicles, no text, neutral presentation.`;
}


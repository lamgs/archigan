// Schema v1 sample projects. Kept only as migration fixtures; the product samples live in samples.ts (v2).
import type { MassingSpec, SiftProject } from "./contracts";
import { normalizeBrief } from "./massing";

// Deterministic v1 massing derivation (the retired pre-v2 generator), kept only to build the fixtures below.
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

const nodes = [
  { id: "brief", type: "brief" as const, position: { x: 60, y: 180 } },
  { id: "massing", type: "massing" as const, position: { x: 350, y: 110 } },
  { id: "refine", type: "refine" as const, position: { x: 650, y: 210 } },
  { id: "export", type: "export" as const, position: { x: 945, y: 135 } },
];

const edges = [
  { id: "brief-massing", source: "brief", target: "massing" },
  { id: "massing-refine", source: "massing", target: "refine" },
  { id: "refine-export", source: "refine", target: "export" },
];

export const legacyDefaultGraph = { nodes, edges };

function sample(id: string, name: string, prompt: string, refinement: string): SiftProject {
  const now = "2026-10-08T12:00:00.000Z";
  return {
    schemaVersion: 1,
    id,
    name,
    createdAt: now,
    updatedAt: now,
    prompt,
    refinement,
    provider: "procedural",
    massing: deriveMassing(prompt, refinement),
    graph: structuredClone(legacyDefaultGraph),
  };
}

export const legacySamples = [
  sample("sample-courtyard", "Courtyard Commons", "A warm terracotta arts center organized around a shaded courtyard", "Step the upper floors into planted terraces"),
  sample("sample-tower", "Spiral Habitat", "A slender glass residential tower with a gentle twist", "Create a generous public base and a rooftop garden"),
  sample("sample-pavilion", "River Archive", "A low concrete archive pavilion stretched along the river", "Carve an atrium void and lift the entry canopy"),
];


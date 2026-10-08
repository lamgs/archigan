// Schema v1 sample projects. Kept only as migration fixtures; the product samples live in samples.ts (v2).
import type { SiftProject } from "./contracts";
import { deriveMassing } from "./massing";

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


import type { SiftProject } from "./contracts";
import { deriveMassing } from "./massing";
import { defaultGraph } from "./samples";

export const EXAMPLE_PROMPTS = [
  { label: "Terraced tower", prompt: "A terraced mixed-use tower rising from a public podium", refinement: "Step the upper floors back into planted terraces" },
  { label: "Twin towers", prompt: "Twin glass towers rising from a shared podium", refinement: "Different heights, crown roofs" },
  { label: "Cylindrical tower", prompt: "A cylindrical glass residential tower", refinement: "" },
  { label: "Twisting tower", prompt: "A slender concrete tower with a spiral twist", refinement: "" },
] as const;

export const MAX_NAME_LENGTH = 80;

export type NameCheck = { ok: true; name: string } | { ok: false; error: string };

export function validateProjectName(raw: string): NameCheck {
  const name = raw.trim().replace(/\s+/g, " ");
  if (!name) return { ok: false, error: "Give the project a name." };
  if (name.length > MAX_NAME_LENGTH) return { ok: false, error: `Project names are limited to ${MAX_NAME_LENGTH} characters.` };
  return { ok: true, name };
}

/** Returns `base`, or `base 2`, `base 3`… so it does not collide (case-insensitively) with `existing`. */
export function uniqueName(base: string, existing: string[]) {
  const taken = new Set(existing.map((name) => name.toLowerCase()));
  if (!taken.has(base.toLowerCase())) return base;
  for (let n = 2; ; n += 1) {
    const candidate = `${base} ${n}`;
    if (!taken.has(candidate.toLowerCase())) return candidate;
  }
}

export function isBlankProject(project: Pick<SiftProject, "prompt">) {
  return project.prompt.trim() === "";
}

/** A new, empty project. It is not persistable until it has a prompt (the project schema requires one). */
export function createBlankProject(id: string, now: string, existingNames: string[]): SiftProject {
  return {
    schemaVersion: 1,
    id,
    name: uniqueName("Untitled study", existingNames),
    createdAt: now,
    updatedAt: now,
    prompt: "",
    refinement: "",
    provider: "procedural",
    massing: deriveMassing("building"),
    graph: structuredClone(defaultGraph),
  };
}

/** Opening a sample creates an independent copy so the bundled sample is never mutated or overwritten. */
export function copyFromSample(sample: SiftProject, id: string, now: string, existingNames: string[]): SiftProject {
  return { ...structuredClone(sample), id, name: uniqueName(sample.name, existingNames), createdAt: now, updatedAt: now };
}

export function renameProject(project: SiftProject, rawName: string, now: string): { ok: true; project: SiftProject } | { ok: false; error: string } {
  const check = validateProjectName(rawName);
  if (!check.ok) return check;
  return { ok: true, project: { ...project, name: check.name, updatedAt: now } };
}

import type { SiftProjectV2 } from "./contracts";
import { runGeneration, type FlowGraph } from "./workflow";

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

export const NODE_ORDER = ["prompt", "generation", "variation", "model", "render"] as const;

export const DEFAULT_VIEWPORT = { x: 40, y: 70, zoom: 0.8 };

function promptText(project: SiftProjectV2) {
  const value = project.graph.nodes.find((node) => node.type === "prompt")?.params.text;
  return typeof value === "string" ? value : "";
}

export function projectPrompt(project: SiftProjectV2) {
  return promptText(project).trim();
}

export function isBlankProject(project: SiftProjectV2) {
  return projectPrompt(project) === "";
}

/** The default four-node workflow: prompt -> generation -> variation -> model. */
export function workflowGraph(prefix: string, prompt: string, refinement: string): FlowGraph {
  return {
    nodes: [
      { id: `${prefix}-prompt`, type: "prompt", position: { x: 40, y: 150 }, params: { text: prompt } },
      { id: `${prefix}-generation`, type: "generation", position: { x: 340, y: 110 }, params: {} },
      { id: `${prefix}-variation`, type: "variation", position: { x: 650, y: 190 }, params: { text: refinement } },
      { id: `${prefix}-model`, type: "model", position: { x: 950, y: 130 }, params: {} },
    ],
    edges: [
      { id: `${prefix}-e1`, source: `${prefix}-prompt`, sourcePort: "prompt", target: `${prefix}-generation`, targetPort: "prompt" },
      { id: `${prefix}-e2`, source: `${prefix}-generation`, sourcePort: "spec", target: `${prefix}-variation`, targetPort: "spec" },
      { id: `${prefix}-e3`, source: `${prefix}-variation`, sourcePort: "spec", target: `${prefix}-model`, targetPort: "spec" },
    ],
  };
}

/** Builds a project from a brief and, when the brief is non-empty, runs its generation node so it opens with a model. */
export function createWorkflowProject(input: { id: string; name: string; now: string; prompt: string; refinement: string }): SiftProjectV2 {
  const base: SiftProjectV2 = {
    schemaVersion: 2,
    id: input.id,
    name: input.name,
    createdAt: input.now,
    updatedAt: input.now,
    viewport: { ...DEFAULT_VIEWPORT },
    graph: workflowGraph(input.id, input.prompt, input.refinement),
    artifacts: {},
    jobs: {},
    revisions: {},
    settings: { provider: "procedural" },
  };
  if (!input.prompt.trim()) return base;
  const run = runGeneration({ ...base.graph, artifacts: {}, jobs: {}, revisions: {} }, `${input.id}-generation`, "procedural", { artifact: `${input.id}-artifact-1`, job: `${input.id}-job-1`, revision: `${input.id}-rev-1` }, input.now);
  if (!run.ok) return base;
  const { nodes, edges, artifacts, jobs, revisions } = run.state;
  return { ...base, graph: { nodes, edges }, artifacts, jobs, revisions };
}

/** A new, empty project. It is not persistable until its prompt node has text. */
export function createBlankProject(id: string, now: string, existingNames: string[]): SiftProjectV2 {
  return createWorkflowProject({ id, name: uniqueName("Untitled study", existingNames), now, prompt: "", refinement: "" });
}

/** Opening a sample creates an independent copy (new id, node/artifact ids rewritten) so the bundled sample is never overwritten. */
export function copyFromSample(sample: SiftProjectV2, id: string, now: string, existingNames: string[]): SiftProjectV2 {
  const brief = projectBrief(sample);
  return createWorkflowProject({ id, name: uniqueName(sample.name, existingNames), now, ...brief });
}

export function renameProject(project: SiftProjectV2, rawName: string, now: string): { ok: true; project: SiftProjectV2 } | { ok: false; error: string } {
  const check = validateProjectName(rawName);
  if (!check.ok) return check;
  return { ok: true, project: { ...project, name: check.name, updatedAt: now } };
}

/**
 * Content signature used to detect unsaved changes. It covers everything that is persisted except timestamps, and is
 * built field by field so key order and `undefined` values never cause false differences.
 */
export function projectSignature(p: Pick<SiftProjectV2, "name" | "viewport" | "settings" | "artifacts" | "jobs" | "revisions"> & { graph: FlowGraph }) {
  const round = (n: number) => Math.round(n * 100) / 100;
  return JSON.stringify([
    p.name,
    [round(p.viewport.x), round(p.viewport.y), round(p.viewport.zoom * 1000) / 1000],
    p.settings.provider,
    p.settings.viewer ?? null,
    p.graph.nodes.map((n) => [n.id, n.type, round(n.position.x), round(n.position.y), n.params, n.artifactId ?? null]),
    p.graph.edges.map((e) => [e.id, e.source, e.sourcePort, e.target, e.targetPort]),
    Object.keys(p.artifacts).sort(),
    Object.values(p.jobs).map((j) => [j.id, j.status, j.resultArtifactId ?? null]).sort(),
    Object.keys(p.revisions).sort(),
  ]);
}

export function projectBrief(project: SiftProjectV2) {
  const refinement = project.graph.nodes.find((node) => node.type === "variation")?.params.text;
  return { prompt: projectPrompt(project), refinement: typeof refinement === "string" ? refinement.trim() : "" };
}

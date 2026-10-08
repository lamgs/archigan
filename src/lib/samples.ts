import { siftProjectV2Schema, type SiftProjectV2 } from "./contracts";
import { createWorkflowProject } from "./projects";
import { addNode, branchFrom, commitVariations, connectNodes, editNodeGeometry, type RunInput } from "./workflow";

const NOW = "2026-10-08T12:00:00.000Z";

const simple = (id: string, name: string, prompt: string, refinement: string): SiftProjectV2 => createWorkflowProject({ id, name, now: NOW, prompt, refinement });

const ok = <T extends { ok: boolean }>(result: T): Extract<T, { ok: true }> => {
  if (!result.ok) throw new Error(`Sample construction failed: ${JSON.stringify(result)}`);
  return result as Extract<T, { ok: true }>;
};

/**
 * The featured portfolio sample: one brief, one generation, two retained branches (a follow-up prompt and a parameter
 * edit) with recorded lineage, and a Render node that renders itself when the sample is opened (`autoRender`), so the
 * image is always produced from the current geometry code and never goes stale.
 */
export function buildTerracedTowerStudy(): SiftProjectV2 {
  const id = "sample-terraced-tower";
  const base = createWorkflowProject({ id, name: "Terraced Tower Study", now: NOW, prompt: "A terraced mixed-use tower rising from a public podium, stepping back in planted terraces", refinement: "" });
  let state: RunInput = { ...base.graph, artifacts: base.artifacts, jobs: base.jobs, revisions: base.revisions };

  const lane = ok(branchFrom(state, `${id}-generation`, { variation: `${id}-variation-b`, model: `${id}-model-b`, edgeA: `${id}-e-b1`, edgeB: `${id}-e-b2` }));
  state = { ...state, nodes: lane.graph.nodes, edges: lane.graph.edges };
  // Branch A — follow-up prompt (a word the local interpreter understands): an all-glass facade.
  state = { ...state, nodes: state.nodes.map((node) => (node.id === `${id}-variation` ? { ...node, params: { ...node.params, text: "glass facade" } } : node)) };
  // Branch B — parameter edits: deeper, more frequent setbacks.
  state = ok(editNodeGeometry(state, `${id}-variation-b`, { op: "setback", id: "tower", every: 3, amount: 2.2 }, { artifact: "unused", revision: "unused" }, NOW)).state;
  state = ok(editNodeGeometry(state, `${id}-variation-b`, { op: "volume", id: "tower", field: "floorCount", value: 16 }, { artifact: "unused", revision: "unused" }, NOW)).state;

  let counter = 0;
  state = commitVariations(state, (prefix) => `${id}-snapshot-${prefix}-${(counter += 1)}`, NOW);

  // A Render node on Branch A, rendered on open.
  state = { ...state, nodes: addNode(state, "render", `${id}-render`, { x: 1290, y: 120 }).nodes };
  state = { ...state, edges: ok(connectNodes(state, { source: `${id}-variation`, sourceHandle: "spec", target: `${id}-render`, targetHandle: "model" }, `${id}-e-render`)).graph.edges };
  state = { ...state, nodes: state.nodes.map((node) => (node.id === `${id}-render` ? { ...node, params: { preset: "axonometric", mode: "shaded", lighting: "studio", background: "paper", resolution: "1600x900", autoRender: true } } : node)) };

  return siftProjectV2Schema.parse({ ...base, viewport: { x: 20, y: 40, zoom: 0.62 }, graph: { nodes: state.nodes, edges: state.edges }, artifacts: state.artifacts, jobs: state.jobs, revisions: state.revisions });
}

export const FEATURED_SAMPLE_ID = "sample-terraced-tower";

/** Short blurbs shown on dashboard cards. */
export const SAMPLE_BLURBS: Record<string, string> = {
  "sample-terraced-tower": "Featured · full board: two branches, lineage, and a render",
  "sample-twin-towers": "Twin towers rising from a shared podium",
  "sample-cylinder": "A cylindrical glass residence with a tapered crown",
  "sample-tower": "A slender glass tower with a gentle twist",
  "sample-pavilion": "A low concrete archive pavilion",
};

export const sampleProjects: SiftProjectV2[] = [
  buildTerracedTowerStudy(),
  simple("sample-twin-towers", "Twin Towers on a Shared Podium", "Twin towers rising from a shared podium", "Glass facade"),
  simple("sample-cylinder", "Cylindrical Residence", "A cylindrical glass residential tower with a rooftop garden", ""),
  simple("sample-tower", "Spiral Habitat", "A slender glass residential tower with a gentle twist", "Create a generous public base and a rooftop garden"),
  simple("sample-pavilion", "River Archive", "A low concrete archive pavilion stretched along the river", ""),
];

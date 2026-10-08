import {
  buildingSpecSchema,
  type Artifact,
  type BuildingSpec,
  type DesignEdge,
  type DesignNode,
  type DesignNodeType,
  type DesignRevision,
  type GenerationJob,
  type Provider,
} from "./contracts";
import { NODE_PORTS, validateConnection } from "./graph";
import { parseRenderSettings, renderInputKey } from "./render-settings";
import { applyEdit, applyEdits, describeEdit, mergeEdit, type SpecEdit } from "./spec-edit";
import { deriveBuildingSpec, normalizeBriefText } from "./typologies";

export type FlowGraph = { nodes: DesignNode[]; edges: DesignEdge[] };
export type Brief = { prompt: string; refinement: string };
export type NodeOutput = { kind: "prompt"; text: string } | { kind: "spec"; spec: BuildingSpec; brief: Brief };
/** `pending` is used by Render nodes whose image is missing or no longer matches the model/settings. */
export type NodeResult = { status: "ready" | "stale" | "pending"; output: NodeOutput; message?: string } | { status: "blocked"; message: string };

export const NODE_LABELS: Record<DesignNodeType, string> = { prompt: "Prompt", generation: "Generation", variation: "Variation", model: "Model", render: "Render" };

const text = (node: DesignNode) => (typeof node.params.text === "string" ? node.params.text : "");
const nodeText = (node: DesignNode) => text(node).trim();
const same = (a: string, b: string) => normalizeBriefText(a) === normalizeBriefText(b);

/** Parameter edits stored on a variation node (replayed over its derived spec; the upstream artifact is never touched). */
export function variationEdits(node: Pick<DesignNode, "params">): SpecEdit[] {
  return Array.isArray(node.params.edits) ? (node.params.edits as SpecEdit[]) : [];
}

export function defaultParams(type: DesignNodeType): Record<string, unknown> {
  return type === "prompt" || type === "variation" ? { text: "" } : {};
}

/** Result of the node feeding `nodeId` through its (single) input, or undefined if nothing is connected. */
function incoming(graph: FlowGraph, nodeId: string) {
  return graph.edges.find((edge) => edge.target === nodeId);
}

/**
 * Pure evaluation of the artifact flow. A generation node only produces output once it has been run (it owns an
 * immutable artifact); everything downstream is derived from that artifact, so wiring has real execution meaning.
 */
export function evaluateGraph(graph: FlowGraph, artifacts: Record<string, Artifact>): Record<string, NodeResult> {
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
  const memo: Record<string, NodeResult> = {};
  const visiting = new Set<string>();

  const evaluate = (id: string): NodeResult => {
    if (memo[id]) return memo[id];
    const node = byId.get(id);
    if (!node) return { status: "blocked", message: "Missing node." };
    if (visiting.has(id)) return { status: "blocked", message: "Cycle detected." };
    visiting.add(id);
    const result = compute(node);
    visiting.delete(id);
    return (memo[id] = result);
  };

  const upstream = (node: DesignNode): NodeResult | undefined => {
    const edge = incoming(graph, node.id);
    return edge ? evaluate(edge.source) : undefined;
  };

  const compute = (node: DesignNode): NodeResult => {
    if (node.type === "prompt") {
      const value = nodeText(node);
      return value ? { status: "ready", output: { kind: "prompt", text: value } } : { status: "blocked", message: "Write a design brief." };
    }
    const input = upstream(node);
    if (!input) return { status: "blocked", message: node.type === "generation" ? "Connect a prompt." : "Connect a model source." };
    if (input.status === "blocked") return input;

    if (node.type === "generation") {
      if (input.output.kind !== "prompt") return { status: "blocked", message: "Generation needs a prompt." };
      const artifact = node.artifactId ? artifacts[node.artifactId] : undefined;
      if (!artifact && node.params.hostedArtifactId) return { status: "blocked", message: "A hosted model exists, but it has no editable geometry. Switch the provider to Local and Run for a parametric design." };
      if (!artifact) return { status: "blocked", message: "Not generated yet — press Run." };
      const parsedSpec = buildingSpecSchema.safeParse(artifact.metadata.spec);
      const brief = artifact.metadata.brief as Brief | undefined;
      if (!brief || !parsedSpec.success) {
        // Artifacts migrated from schema v1 carry only an approximate spec; derive the canonical one from the prompt.
        const migrated: Brief = { prompt: input.output.text, refinement: "" };
        return { status: "ready", output: { kind: "spec", spec: deriveBuildingSpec(migrated.prompt, migrated.refinement), brief: migrated } };
      }
      return { status: same(brief.prompt, input.output.text) ? "ready" : "stale", output: { kind: "spec", spec: parsedSpec.data, brief } };
    }

    if (input.output.kind !== "spec") return { status: "blocked", message: "Needs a generated model." };
    if (node.type === "variation") {
      const extra = nodeText(node);
      const brief: Brief = { prompt: input.output.brief.prompt, refinement: [input.output.brief.refinement, extra].filter(Boolean).join(" ") };
      const base = extra ? deriveBuildingSpec(brief.prompt, brief.refinement) : input.output.spec;
      const spec = applyEdits(base, variationEdits(node));
      return { status: input.status, output: { kind: "spec", spec, brief } };
    }
    if (node.type === "render") {
      const artifact = node.artifactId ? artifacts[node.artifactId] : undefined;
      const key = renderInputKey(input.output.spec, parseRenderSettings(node.params));
      if (!artifact) return { status: "pending", output: input.output, message: "Not rendered yet — press Render." };
      if (artifact.metadata.inputKey !== key) return { status: "pending", output: input.output, message: "The model or settings changed — render again." };
      return { status: input.status, output: input.output };
    }
    return { status: input.status, output: input.output }; // model nodes display the spec
  };

  graph.nodes.forEach((node) => evaluate(node.id));
  return memo;
}

export type Mutation<T> = ({ ok: true } & T) | { ok: false; message: string };

export type ConnectionInput = { source: string; sourceHandle?: string | null; target: string; targetHandle?: string | null };

export function connectNodes(graph: FlowGraph, connection: ConnectionInput, edgeId: string): Mutation<{ graph: FlowGraph }> {
  const candidate = { source: connection.source, sourcePort: connection.sourceHandle ?? "", target: connection.target, targetPort: connection.targetHandle ?? "" };
  const check = validateConnection(graph.nodes, graph.edges, candidate);
  if (!check.ok) return { ok: false, message: check.message };
  return { ok: true, graph: { ...graph, edges: [...graph.edges, { id: edgeId, ...candidate }] } };
}

/** Node types that can legally be attached to an output of `type`. */
export function nextNodeTypes(type: DesignNodeType): DesignNodeType[] {
  const outputs = new Set(NODE_PORTS[type].outputs.map((port) => port.kind));
  return (Object.keys(NODE_PORTS) as DesignNodeType[]).filter((candidate) => NODE_PORTS[candidate].inputs.some((port) => outputs.has(port.kind)));
}

export function addNode(graph: FlowGraph, type: DesignNodeType, id: string, position: { x: number; y: number }): FlowGraph {
  return { ...graph, nodes: [...graph.nodes, { id, type, position, params: defaultParams(type) }] };
}

/** Adds a node of `type` and wires it to the first compatible free output of `sourceId`. */
export function addConnectedNode(graph: FlowGraph, sourceId: string, type: DesignNodeType, ids: { node: string; edge: string }): Mutation<{ graph: FlowGraph }> {
  const source = graph.nodes.find((node) => node.id === sourceId);
  if (!source) return { ok: false, message: "Select a node first." };
  const added = addNode(graph, type, ids.node, { x: source.position.x + 300, y: source.position.y + 20 * graph.edges.filter((edge) => edge.source === sourceId).length });
  for (const out of NODE_PORTS[source.type].outputs) {
    for (const input of NODE_PORTS[type].inputs) {
      if (out.kind !== input.kind) continue;
      const result = connectNodes(added, { source: sourceId, sourceHandle: out.id, target: ids.node, targetHandle: input.id }, ids.edge);
      if (result.ok) return result;
    }
  }
  return { ok: false, message: `${NODE_LABELS[type]} cannot follow ${NODE_LABELS[source.type]}.` };
}

export type RunInput = { nodes: DesignNode[]; edges: DesignEdge[]; artifacts: Record<string, Artifact>; jobs: Record<string, GenerationJob>; revisions: Record<string, DesignRevision> };

/**
 * Runs a generation node: creates a new immutable building-spec artifact and a completed job. A previous artifact is
 * kept and linked through a revision rather than being overwritten.
 */
export function runGeneration(state: RunInput, nodeId: string, provider: Provider, ids: { artifact: string; job: string; revision: string }, now: string): Mutation<{ state: RunInput }> {
  const node = state.nodes.find((item) => item.id === nodeId);
  if (!node || node.type !== "generation") return { ok: false, message: "Select a generation node to run." };
  if (provider !== "procedural") return { ok: false, message: "Hosted generation is not available yet; switch to Local." };
  if (ids.artifact in state.artifacts || ids.job in state.jobs || ids.revision in state.revisions) return { ok: false, message: "Generated ids collide with existing records; try again." };
  const input = incoming(state, nodeId);
  const source = input && evaluateGraph(state, state.artifacts)[input.source];
  if (!source) return { ok: false, message: "Connect a prompt to this generation node." };
  if (source.status === "blocked") return { ok: false, message: source.message };
  if (source.output.kind !== "prompt") return { ok: false, message: "Generation needs a prompt." };

  const brief: Brief = { prompt: source.output.text, refinement: "" };
  const spec = deriveBuildingSpec(brief.prompt, brief.refinement);
  const artifact: Artifact = { id: ids.artifact, kind: "building-spec", sourceNodeId: nodeId, createdAt: now, storageKey: `inline:${ids.artifact}`, metadata: { spec, brief, origin: "procedural" } };
  const job: GenerationJob = { id: ids.job, nodeId, provider, status: "completed", resultArtifactId: artifact.id };
  const revisions = node.artifactId && state.artifacts[node.artifactId]
    ? { ...state.revisions, [ids.revision]: { id: ids.revision, parentArtifactId: node.artifactId, childArtifactId: artifact.id, sourceNodeIds: [nodeId], change: "prompt" as const, instruction: brief.prompt, createdAt: now } }
    : state.revisions;
  return {
    ok: true,
    state: { ...state, nodes: state.nodes.map((item) => (item.id === nodeId ? { ...item, artifactId: artifact.id } : item)), artifacts: { ...state.artifacts, [artifact.id]: artifact }, jobs: { ...state.jobs, [job.id]: job }, revisions },
  };
}

/** The spec the viewer should show: the selected node's output, else the first model node, else any spec. */
export function previewSpec(results: Record<string, NodeResult>, graph: FlowGraph, selectedId?: string): { spec: BuildingSpec; nodeId: string; stale: boolean } | undefined {
  const pick = (id: string | undefined) => {
    const result = id ? results[id] : undefined;
    return id && result && result.status !== "blocked" && result.output.kind === "spec" ? { spec: result.output.spec, nodeId: id, stale: result.status === "stale" } : undefined;
  };
  return pick(selectedId) ?? pick(graph.nodes.find((node) => node.type === "model" && pick(node.id))?.id) ?? pick(graph.nodes.find((node) => pick(node.id))?.id);
}

/**
 * Applies a geometry edit to the spec a node outputs without overwriting its source:
 * - variation node: the edit is stored on the node and replayed over the upstream spec;
 * - generation node: a new child artifact is created, the old one is kept, and a `parameters` revision links them.
 */
export function editNodeGeometry(state: RunInput, nodeId: string, edit: SpecEdit, ids: { artifact: string; revision: string }, now: string): Mutation<{ state: RunInput }> {
  const node = state.nodes.find((item) => item.id === nodeId);
  if (!node) return { ok: false, message: "Select a node first." };
  const results = evaluateGraph(state, state.artifacts);
  const current = results[nodeId];
  if (!current || current.status === "blocked" || current.output.kind !== "spec") return { ok: false, message: "This node has no model to edit yet." };
  if (node.type === "generation" && (ids.artifact in state.artifacts || ids.revision in state.revisions)) return { ok: false, message: "Generated ids collide with existing records; try again." };

  if (node.type === "variation") {
    const edits = mergeEdit(variationEdits(node), edit);
    const trial = { ...state, nodes: state.nodes.map((item) => (item.id === nodeId ? { ...item, params: { ...item.params, edits } } : item)) };
    const check = applyEdit(current.output.spec, edit); // validate against what the user is looking at
    if (!check.ok) return check;
    return { ok: true, state: trial };
  }
  if (node.type !== "generation" || !node.artifactId || !state.artifacts[node.artifactId]) return { ok: false, message: "Geometry can be edited on Generation and Variation nodes." };
  const result = applyEdit(current.output.spec, edit);
  if (!result.ok) return result;
  const parent = state.artifacts[node.artifactId];
  const artifact: Artifact = { id: ids.artifact, kind: "building-spec", sourceNodeId: nodeId, createdAt: now, storageKey: `inline:${ids.artifact}`, metadata: { spec: result.spec, brief: current.output.brief, origin: "parameters", edit, parentArtifactId: parent.id } };
  const revision: DesignRevision = { id: ids.revision, parentArtifactId: parent.id, childArtifactId: artifact.id, sourceNodeIds: [nodeId], change: "parameters", instruction: describeEdit(edit), createdAt: now };
  return { ok: true, state: { ...state, nodes: state.nodes.map((item) => (item.id === nodeId ? { ...item, artifactId: artifact.id } : item)), artifacts: { ...state.artifacts, [artifact.id]: artifact }, revisions: { ...state.revisions, [revision.id]: revision } } };
}

// ---------------------------------------------------------------------------
// Branching and lineage (P0.14)
// ---------------------------------------------------------------------------

export const branchLabel = (index: number) => `Branch ${String.fromCharCode(65 + (index % 26))}`;

/** The nearest artifact feeding `nodeId` through its input chain (generation/variation snapshots). */
export function upstreamArtifactId(graph: FlowGraph, nodeId: string): string | undefined {
  const seen = new Set<string>();
  let current = incoming(graph, nodeId)?.source;
  while (current && !seen.has(current)) {
    seen.add(current);
    const node = graph.nodes.find((item) => item.id === current);
    if (node?.artifactId) return node.artifactId;
    current = incoming(graph, current)?.source;
  }
  return undefined;
}

/**
 * Forks a design: adds a sibling Variation + Model lane after `sourceId`, wired from the same output, so both
 * branches stay visible with the shared source as their common parent. Existing lanes are labelled A, B, …
 */
export function branchFrom(graph: FlowGraph, sourceId: string, ids: { variation: string; model: string; edgeA: string; edgeB: string }): Mutation<{ graph: FlowGraph; variationId: string }> {
  const source = graph.nodes.find((node) => node.id === sourceId);
  if (!source || (source.type !== "generation" && source.type !== "variation")) return { ok: false, message: "Branch from a Generation or Variation node." };
  const siblings = graph.edges.filter((edge) => edge.source === sourceId).map((edge) => graph.nodes.find((node) => node.id === edge.target)).filter((node): node is DesignNode => node?.type === "variation");
  const labelled = graph.nodes.map((node) => {
    const index = siblings.findIndex((sibling) => sibling.id === node.id);
    return index >= 0 && !node.params.label ? { ...node, params: { ...node.params, label: branchLabel(index) } } : node;
  });
  const lowest = Math.max(source.position.y, ...siblings.map((sibling) => sibling.position.y));
  const added = addNode(addNode({ ...graph, nodes: labelled }, "variation", ids.variation, { x: source.position.x + 300, y: lowest + 230 }), "model", ids.model, { x: source.position.x + 600, y: lowest + 200 });
  const withLabel = { ...added, nodes: added.nodes.map((node) => (node.id === ids.variation ? { ...node, params: { ...node.params, label: branchLabel(siblings.length) } } : node)) };
  const first = connectNodes(withLabel, { source: sourceId, sourceHandle: "spec", target: ids.variation, targetHandle: "spec" }, ids.edgeA);
  if (!first.ok) return first;
  const second = connectNodes(first.graph, { source: ids.variation, sourceHandle: "spec", target: ids.model, targetHandle: "spec" }, ids.edgeB);
  return second.ok ? { ok: true, graph: second.graph, variationId: ids.variation } : second;
}

const recipeKey = (node: DesignNode, parent: string | undefined) => JSON.stringify({ text: nodeText(node), edits: variationEdits(node), parent });

/** Asks the id generator for ids until one is unused, so an immutable record can never be overwritten by a collision. */
function uniqueId(newId: (prefix: string) => string, prefix: string, taken: (candidate: string) => boolean) {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    const candidate = newId(prefix);
    if (!taken(candidate)) return candidate;
  }
  throw new Error(`Could not generate an unused ${prefix} id.`);
}

/**
 * Snapshots every Variation node whose recipe (follow-up text + parameter edits, over its current parent artifact)
 * changed since its last snapshot: a new immutable artifact plus a revision from the parent artifact.
 * Pass-through variations (no text, no edits) are not snapshotted.
 */
export function commitVariations(state: RunInput, newId: (prefix: string) => string, now: string): RunInput {
  let next = state;
  state.nodes.filter((node) => node.type === "variation").forEach((node) => {
    const current = next.nodes.find((item) => item.id === node.id) as DesignNode;
    const edits = variationEdits(current);
    const text = nodeText(current);
    if (!text && edits.length === 0) return;
    const parent = upstreamArtifactId(next, node.id);
    const key = recipeKey(current, parent);
    if (current.artifactId && next.artifacts[current.artifactId]?.metadata.recipeKey === key) return;
    const result = evaluateGraph(next, next.artifacts)[node.id];
    if (!result || result.status === "blocked" || result.output.kind !== "spec" || !parent) return;
    const artifactId = uniqueId(newId, "artifact", (candidate) => candidate in next.artifacts);
    const artifact: Artifact = { id: artifactId, kind: "building-spec", sourceNodeId: node.id, createdAt: now, storageKey: `inline:${artifactId}`, metadata: { spec: result.output.spec, brief: result.output.brief, origin: "variation", recipeKey: key, parentArtifactId: parent } };
    const revisionId = uniqueId(newId, "rev", (candidate) => candidate in next.revisions);
    const revision: DesignRevision = { id: revisionId, parentArtifactId: parent, childArtifactId: artifactId, sourceNodeIds: [node.id], change: text ? "prompt" : "parameters", instruction: [text, ...edits.map(describeEdit)].filter(Boolean).join("; ").slice(0, 800), createdAt: now };
    next = { ...next, nodes: next.nodes.map((item) => (item.id === node.id ? { ...item, artifactId } : item)), artifacts: { ...next.artifacts, [artifactId]: artifact }, revisions: { ...next.revisions, [revisionId]: revision } };
  });
  return next;
}

export type LineageEntry = { artifactId: string; createdAt: string; origin: string; instruction: string; current: boolean };

/** Root-to-leaf chain of artifacts ending at `artifactId`, following revisions back to the original generation. */
export function lineageOf(revisions: Record<string, DesignRevision>, artifacts: Record<string, Artifact>, artifactId: string | undefined): LineageEntry[] {
  const chain: LineageEntry[] = [];
  const seen = new Set<string>();
  let current = artifactId;
  let instruction = "";
  while (current && artifacts[current] && !seen.has(current)) {
    seen.add(current);
    const artifact = artifacts[current];
    chain.unshift({ artifactId: current, createdAt: artifact.createdAt, origin: String(artifact.metadata.origin ?? "generation"), instruction, current: current === artifactId });
    const revision = Object.values(revisions).find((item) => item.childArtifactId === current);
    instruction = revision?.instruction ?? "";
    current = revision?.parentArtifactId;
  }
  // instruction describes the step that *produced* each entry, so shift it down by one position.
  return chain.map((entry, index) => ({ ...entry, instruction: index === 0 ? "Original generation" : Object.values(revisions).find((item) => item.childArtifactId === entry.artifactId)?.instruction ?? "" }));
}

/** Every artifact a node has produced, oldest first, with the instruction of the revision that created it. */
export function versionsOf(state: Pick<RunInput, "artifacts" | "revisions">, nodeId: string, currentId?: string): LineageEntry[] {
  return Object.values(state.artifacts)
    .filter((artifact) => artifact.sourceNodeId === nodeId)
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt) || a.id.localeCompare(b.id))
    .map((artifact) => ({
      artifactId: artifact.id,
      createdAt: artifact.createdAt,
      origin: String(artifact.metadata.origin ?? "generation"),
      instruction: Object.values(state.revisions).find((revision) => revision.childArtifactId === artifact.id)?.instruction || "Original generation",
      current: artifact.id === currentId,
    }));
}

/** Points a Generation node at one of its own earlier artifacts (a pointer move; nothing is deleted). */
export function restoreVersion(state: RunInput, nodeId: string, artifactId: string): Mutation<{ state: RunInput }> {
  const node = state.nodes.find((item) => item.id === nodeId);
  if (!node || node.type !== "generation") return { ok: false, message: "Only Generation nodes can switch versions." };
  if (state.artifacts[artifactId]?.sourceNodeId !== nodeId) return { ok: false, message: "That version was not produced by this node." };
  return { ok: true, state: { ...state, nodes: state.nodes.map((item) => (item.id === nodeId ? { ...item, artifactId } : item)) } };
}

// ---------------------------------------------------------------------------
// Render artifacts (P0.16)
// ---------------------------------------------------------------------------

export type RenderRecord = { artifactId: string; width: number; height: number; bytes: number };

/**
 * Records a finished render as an immutable `render-png` artifact bound to the exact model and settings it was made
 * from. The previous render stays in `artifacts`; the node simply points at the newest one.
 */
export function recordRender(state: RunInput, nodeId: string, record: RenderRecord, now: string): Mutation<{ state: RunInput }> {
  const node = state.nodes.find((item) => item.id === nodeId);
  if (!node || node.type !== "render") return { ok: false, message: "Select a Render node." };
  const result = evaluateGraph(state, state.artifacts)[nodeId];
  if (!result || result.status === "blocked" || result.output.kind !== "spec") return { ok: false, message: "Connect a generated model to this Render node first." };
  const settings = parseRenderSettings(node.params);
  const artifact: Artifact = {
    id: record.artifactId,
    kind: "render-png",
    sourceNodeId: nodeId,
    createdAt: now,
    storageKey: `asset:${record.artifactId}`,
    metadata: { origin: "render", settings, width: record.width, height: record.height, bytes: record.bytes, inputKey: renderInputKey(result.output.spec, settings), parentArtifactId: upstreamArtifactId(state, nodeId) ?? null },
  };
  return { ok: true, state: { ...state, nodes: state.nodes.map((item) => (item.id === nodeId ? { ...item, artifactId: artifact.id } : item)), artifacts: { ...state.artifacts, [artifact.id]: artifact } } };
}

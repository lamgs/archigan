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
import { applyEdit, applyEdits, describeEdit, mergeEdit, type SpecEdit } from "./spec-edit";
import { deriveBuildingSpec, normalizeBriefText } from "./typologies";

export type FlowGraph = { nodes: DesignNode[]; edges: DesignEdge[] };
export type Brief = { prompt: string; refinement: string };
export type NodeOutput = { kind: "prompt"; text: string } | { kind: "spec"; spec: BuildingSpec; brief: Brief };
export type NodeResult = { status: "ready" | "stale"; output: NodeOutput } | { status: "blocked"; message: string };

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
    return { status: input.status, output: input.output }; // model / render consume the spec
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

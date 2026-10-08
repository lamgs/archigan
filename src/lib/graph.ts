import type { DesignEdge, DesignNode, DesignNodeType } from "./contracts";

export type PortKind = "prompt" | "building-spec" | "model-glb" | "render-png";
export type PortDef = { id: string; kind: PortKind };
export type NodePorts = { inputs: PortDef[]; outputs: PortDef[] };

export const NODE_PORTS: Record<DesignNodeType, NodePorts> = {
  prompt: { inputs: [], outputs: [{ id: "prompt", kind: "prompt" }] },
  generation: { inputs: [{ id: "prompt", kind: "prompt" }], outputs: [{ id: "spec", kind: "building-spec" }] },
  variation: { inputs: [{ id: "spec", kind: "building-spec" }], outputs: [{ id: "spec", kind: "building-spec" }] },
  model: { inputs: [{ id: "spec", kind: "building-spec" }], outputs: [{ id: "glb", kind: "model-glb" }] },
  render: { inputs: [{ id: "model", kind: "building-spec" }], outputs: [{ id: "image", kind: "render-png" }] },
};

export type ConnectionCheck = { ok: true } | { ok: false; code: ConnectionErrorCode; message: string };
export type ConnectionErrorCode = "missing-node" | "unknown-port" | "self-loop" | "type-mismatch" | "duplicate" | "input-occupied" | "cycle";

const fail = (code: ConnectionErrorCode, message: string): ConnectionCheck => ({ ok: false, code, message });

function reaches(edges: DesignEdge[], from: string, to: string) {
  const stack = [from];
  const seen = new Set<string>();
  while (stack.length) {
    const current = stack.pop() as string;
    if (current === to) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    edges.forEach((edge) => edge.source === current && stack.push(edge.target));
  }
  return false;
}

/** Checks whether `candidate` may be added to a graph whose existing edges are assumed valid. */
export function validateConnection(nodes: Pick<DesignNode, "id" | "type">[], edges: DesignEdge[], candidate: Omit<DesignEdge, "id">): ConnectionCheck {
  const source = nodes.find((node) => node.id === candidate.source);
  const target = nodes.find((node) => node.id === candidate.target);
  if (!source || !target) return fail("missing-node", "Connection references a node that does not exist.");
  if (source.id === target.id) return fail("self-loop", "A node cannot connect to itself.");
  const out = NODE_PORTS[source.type].outputs.find((port) => port.id === candidate.sourcePort);
  const input = NODE_PORTS[target.type].inputs.find((port) => port.id === candidate.targetPort);
  if (!out || !input) return fail("unknown-port", `No such port on ${!out ? source.type : target.type} node.`);
  if (out.kind !== input.kind) return fail("type-mismatch", `Cannot connect ${out.kind} output to ${input.kind} input.`);
  if (edges.some((edge) => edge.source === candidate.source && edge.sourcePort === candidate.sourcePort && edge.target === candidate.target && edge.targetPort === candidate.targetPort)) return fail("duplicate", "These ports are already connected.");
  if (edges.some((edge) => edge.target === candidate.target && edge.targetPort === candidate.targetPort)) return fail("input-occupied", "That input already has a source.");
  if (reaches(edges, candidate.target, candidate.source)) return fail("cycle", "This connection would create a cycle.");
  return { ok: true };
}

/** Validates a whole graph by replaying its edges in order. Returns the first problem per rejected edge. */
export function validateGraph(nodes: Pick<DesignNode, "id" | "type">[], edges: DesignEdge[]) {
  const accepted: DesignEdge[] = [];
  const errors: { edgeId: string; code: ConnectionErrorCode; message: string }[] = [];
  edges.forEach((edge) => {
    const result = validateConnection(nodes, accepted, edge);
    if (result.ok) accepted.push(edge);
    else errors.push({ edgeId: edge.id, code: result.code, message: result.message });
  });
  return { ok: errors.length === 0, errors, accepted };
}

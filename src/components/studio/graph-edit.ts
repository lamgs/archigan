import type { FlowGraph } from "@/lib/workflow";
import { validateGraph } from "@/lib/graph";

export type RemovalResult = { ok: true; graph: FlowGraph } | { ok: false; message: string };

/**
 * Removes a node and every connection touching it. Only the board changes: artifacts, jobs and revisions live outside
 * the graph and stay as append-only history (ADR-013). The remaining graph is re-validated by the domain layer.
 */
export function removeNode(graph: FlowGraph, nodeId: string): RemovalResult {
  if (!graph.nodes.some((node) => node.id === nodeId)) return { ok: false, message: "That node no longer exists." };
  const nodes = graph.nodes.filter((node) => node.id !== nodeId);
  const edges = graph.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId);
  return validateGraph(nodes, edges).ok ? { ok: true, graph: { nodes, edges } } : { ok: false, message: "The graph would become invalid, so nothing was removed." };
}

/** Removes one connection; the nodes (and their artifacts) are untouched. */
export function removeEdge(graph: FlowGraph, edgeId: string): RemovalResult {
  if (!graph.edges.some((edge) => edge.id === edgeId)) return { ok: false, message: "That connection no longer exists." };
  const edges = graph.edges.filter((edge) => edge.id !== edgeId);
  return validateGraph(graph.nodes, edges).ok ? { ok: true, graph: { nodes: graph.nodes, edges } } : { ok: false, message: "The graph would become invalid, so nothing was removed." };
}

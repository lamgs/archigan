import { describe, expect, it } from "vitest";
import { validateGraph } from "@/lib/graph";
import type { FlowGraph } from "@/lib/workflow";
import { removeEdge, removeNode } from "./graph-edit";

const node = (id: string, type: "prompt" | "generation" | "model" | "render") => ({ id, type, position: { x: 0, y: 0 }, params: {} });
const graph: FlowGraph = {
  nodes: [node("p", "prompt"), node("g", "generation"), node("m", "model"), node("r", "render")],
  edges: [
    { id: "e1", source: "p", sourcePort: "prompt", target: "g", targetPort: "prompt" },
    { id: "e2", source: "g", sourcePort: "spec", target: "m", targetPort: "spec" },
    { id: "e3", source: "g", sourcePort: "spec", target: "r", targetPort: "model" },
  ],
};

describe("graph removal", () => {
  it("starts from a valid fixture", () => expect(validateGraph(graph.nodes, graph.edges).ok).toBe(true));

  it("removes a node together with every connection that touches it, and stays valid", () => {
    const result = removeNode(graph, "g");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.graph.nodes.map((n) => n.id)).toEqual(["p", "m", "r"]);
    expect(result.graph.edges).toEqual([]);
    expect(validateGraph(result.graph.nodes, result.graph.edges).ok).toBe(true);
  });

  it("removes one connection and keeps both nodes", () => {
    const result = removeEdge(graph, "e2");
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.graph.nodes).toHaveLength(4);
    expect(result.graph.edges.map((e) => e.id)).toEqual(["e1", "e3"]);
  });

  it("does not mutate its input and rejects unknown ids", () => {
    const before = structuredClone(graph);
    removeNode(graph, "g"); removeEdge(graph, "e1");
    expect(graph).toEqual(before);
    expect(removeNode(graph, "nope").ok).toBe(false);
    expect(removeEdge(graph, "nope").ok).toBe(false);
  });
});

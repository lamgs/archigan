import { describe, expect, it } from "vitest";
import { siftProjectV2Schema } from "./contracts";
import { createWorkflowProject, workflowGraph } from "./projects";
import { addConnectedNode, addNode, connectNodes, evaluateGraph, nextNodeTypes, previewSpec, runGeneration, type RunInput } from "./workflow";

const NOW = "2026-10-09T10:00:00.000Z";
const ids = (n: number) => ({ artifact: `a${n}`, job: `j${n}`, revision: `r${n}` });
const empty = { artifacts: {}, jobs: {}, revisions: {} };
const state = (prompt: string, refinement = ""): RunInput => ({ ...workflowGraph("p", prompt, refinement), ...empty });

describe("evaluateGraph", () => {
  it("blocks downstream nodes until the generation node is run", () => {
    const s = state("A terraced tower");
    const before = evaluateGraph(s, s.artifacts);
    expect(before["p-prompt"].status).toBe("ready");
    expect(before["p-generation"]).toMatchObject({ status: "blocked", message: expect.stringMatching(/press Run/) });
    expect(before["p-model"].status).toBe("blocked");
    const run = runGeneration(s, "p-generation", "procedural", ids(1), NOW);
    if (!run.ok) throw new Error(run.message);
    const after = evaluateGraph(run.state, run.state.artifacts);
    expect(after["p-model"].status).toBe("ready");
  });

  it("reports missing prompt text and missing connections", () => {
    const s = state("");
    expect(evaluateGraph(s, {})["p-prompt"]).toMatchObject({ status: "blocked", message: expect.stringMatching(/brief/) });
    expect(evaluateGraph(s, {})["p-generation"].status).toBe("blocked");
    const lonely = { nodes: addNode({ nodes: [], edges: [] }, "model", "m", { x: 0, y: 0 }).nodes, edges: [] };
    expect(evaluateGraph(lonely, {}).m).toMatchObject({ status: "blocked", message: expect.stringMatching(/Connect/) });
  });

  it("marks outputs stale when the upstream prompt changes after a run", () => {
    const run = runGeneration(state("A terraced tower"), "p-generation", "procedural", ids(1), NOW);
    if (!run.ok) throw new Error(run.message);
    const edited = { ...run.state, nodes: run.state.nodes.map((n) => (n.type === "prompt" ? { ...n, params: { text: "Twin towers" } } : n)) };
    const result = evaluateGraph(edited, edited.artifacts);
    expect(result["p-generation"].status).toBe("stale");
    expect(result["p-model"].status).toBe("stale");
  });

  it("applies the variation refinement downstream without altering the generated artifact", () => {
    const run = runGeneration(state("A tower", "glass"), "p-generation", "procedural", ids(1), NOW);
    if (!run.ok) throw new Error(run.message);
    const results = evaluateGraph(run.state, run.state.artifacts);
    const generated = results["p-generation"];
    const model = results["p-model"];
    if (generated.status === "blocked" || model.status === "blocked" || generated.output.kind !== "spec" || model.output.kind !== "spec") throw new Error("expected specs");
    expect(model.output.spec).not.toEqual(generated.output.spec);
    expect(model.output.brief.refinement).toBe("glass");
    expect(run.state.artifacts.a1.metadata.brief).toEqual({ prompt: "A tower", refinement: "" });
  });
});

describe("runGeneration", () => {
  it("keeps the previous artifact and links a revision when run again", () => {
    const first = runGeneration(state("A tower"), "p-generation", "procedural", ids(1), NOW);
    if (!first.ok) throw new Error(first.message);
    const second = runGeneration(first.state, "p-generation", "procedural", ids(2), NOW);
    if (!second.ok) throw new Error(second.message);
    expect(Object.keys(second.state.artifacts).sort()).toEqual(["a1", "a2"]);
    expect(second.state.revisions.r2).toMatchObject({ parentArtifactId: "a1", childArtifactId: "a2" });
    expect(second.state.jobs.j2).toMatchObject({ status: "completed", resultArtifactId: "a2" });
    expect(second.state.nodes.find((n) => n.id === "p-generation")?.artifactId).toBe("a2");
  });
  it("refuses hosted providers, empty prompts, and non-generation nodes", () => {
    expect(runGeneration(state("A tower"), "p-generation", "meshy", ids(1), NOW)).toMatchObject({ ok: false });
    expect(runGeneration(state(""), "p-generation", "procedural", ids(1), NOW)).toMatchObject({ ok: false });
    expect(runGeneration(state("A tower"), "p-model", "procedural", ids(1), NOW)).toMatchObject({ ok: false });
  });
  it("produces a project that satisfies the persisted schema", () => {
    expect(siftProjectV2Schema.safeParse(createWorkflowProject({ id: "x", name: "X", now: NOW, prompt: "A tower", refinement: "" })).success).toBe(true);
  });
});

describe("wiring", () => {
  const g = workflowGraph("p", "A tower", "");
  it("accepts compatible connections and rejects unsupported or cyclic ones", () => {
    const withRender = addNode(g, "render", "r", { x: 0, y: 0 });
    expect(connectNodes(withRender, { source: "p-variation", sourceHandle: "spec", target: "r", targetHandle: "model" }, "e")).toMatchObject({ ok: true });
    expect(connectNodes(withRender, { source: "p-prompt", sourceHandle: "prompt", target: "r", targetHandle: "model" }, "e")).toMatchObject({ ok: false, message: expect.stringMatching(/Cannot connect/) });
    expect(connectNodes(g, { source: "p-model", sourceHandle: "glb", target: "p-generation", targetHandle: "prompt" }, "e")).toMatchObject({ ok: false });
    const loop = addNode(addNode(g, "variation", "v1", { x: 0, y: 0 }), "variation", "v2", { x: 0, y: 0 });
    const a = connectNodes(loop, { source: "v1", sourceHandle: "spec", target: "v2", targetHandle: "spec" }, "e1");
    if (!a.ok) throw new Error(a.message);
    expect(connectNodes(a.graph, { source: "v2", sourceHandle: "spec", target: "v1", targetHandle: "spec" }, "e2")).toMatchObject({ ok: false, message: expect.stringMatching(/cycle/) });
  });
  it("offers only compatible next nodes and auto-wires them", () => {
    expect(nextNodeTypes("prompt")).toEqual(["generation"]);
    expect(nextNodeTypes("generation").sort()).toEqual(["model", "render", "variation"]);
    expect(nextNodeTypes("render")).toEqual([]);
    const added = addConnectedNode(g, "p-generation", "render", { node: "r", edge: "e" });
    expect(added).toMatchObject({ ok: true });
    if (added.ok) expect(added.graph.edges.at(-1)).toMatchObject({ source: "p-generation", sourcePort: "spec", target: "r", targetPort: "model" });
    expect(addConnectedNode(g, "p-model", "render", { node: "r", edge: "e" })).toMatchObject({ ok: false });
  });
});

describe("previewSpec", () => {
  it("prefers the selected node, then a model node", () => {
    const run = runGeneration(state("A tower", "glass"), "p-generation", "procedural", ids(1), NOW);
    if (!run.ok) throw new Error(run.message);
    const results = evaluateGraph(run.state, run.state.artifacts);
    expect(previewSpec(results, run.state, "p-generation")?.nodeId).toBe("p-generation");
    expect(previewSpec(results, run.state, "p-prompt")?.nodeId).toBe("p-model");
    expect(previewSpec(evaluateGraph(state("A tower"), {}), state("A tower"))).toBeUndefined();
  });
});

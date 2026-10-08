import { describe, expect, it } from "vitest";
import { siftProjectV2Schema } from "./contracts";
import { createWorkflowProject, workflowGraph } from "./projects";
import { branchFrom, commitVariations, editNodeGeometry, evaluateGraph, lineageOf, restoreVersion, runGeneration, versionsOf, type RunInput } from "./workflow";

const NOW = "2026-10-09T10:00:00.000Z";
let n = 0;
const newId = (prefix: string) => `${prefix}-${(n += 1)}`;
const ok = <T extends { ok: boolean }>(r: T) => { if (!r.ok) throw new Error(JSON.stringify(r)); return r as Extract<T, { ok: true }>; };
const generated = (refinement = ""): RunInput => ok(runGeneration({ ...workflowGraph("p", "A terraced stepped tower with a podium", refinement), artifacts: {}, jobs: {}, revisions: {} }, "p-generation", "procedural", { artifact: "a1", job: "j1", revision: "r0" }, NOW)).state;
const withVariationText = (s: RunInput, id: string, text: string): RunInput => ({ ...s, nodes: s.nodes.map((node) => (node.id === id ? { ...node, params: { ...node.params, text } } : node)) });
const spec = (s: RunInput, id: string) => { const r = evaluateGraph(s, s.artifacts)[id]; if (r.status === "blocked" || r.output.kind !== "spec") throw new Error("blocked"); return r.output.spec; };

describe("branchFrom", () => {
  const ids = { variation: "v2", model: "m2", edgeA: "ea", edgeB: "eb" };
  it("adds a sibling lane from the same source and labels both branches", () => {
    const state = generated();
    const result = ok(branchFrom(state, "p-generation", ids));
    const edgesFromSource = result.graph.edges.filter((e) => e.source === "p-generation");
    expect(edgesFromSource.map((e) => e.target).sort()).toEqual(["p-variation", "v2"]);
    expect(result.graph.edges.find((e) => e.source === "v2")?.target).toBe("m2");
    expect(result.graph.nodes.find((x) => x.id === "p-variation")?.params.label).toBe("Branch A");
    expect(result.graph.nodes.find((x) => x.id === "v2")?.params.label).toBe("Branch B");
    const [a, b] = ["p-variation", "v2"].map((id) => result.graph.nodes.find((x) => x.id === id)!.position);
    expect(b.y).toBeGreaterThan(a.y); // lanes do not overlap
  });
  it("branches repeatedly and refuses non-branchable nodes", () => {
    const first = ok(branchFrom(generated(), "p-generation", ids));
    const second = ok(branchFrom(first.graph, "p-generation", { variation: "v3", model: "m3", edgeA: "ec", edgeB: "ed" }));
    expect(second.graph.nodes.find((x) => x.id === "v3")?.params.label).toBe("Branch C");
    expect(branchFrom(generated(), "p-model", ids)).toMatchObject({ ok: false });
    expect(branchFrom(generated(), "p-prompt", ids)).toMatchObject({ ok: false });
  });
});

describe("two branches with lineage", () => {
  const build = () => {
    let state = generated();
    const branched = ok(branchFrom(state, "p-generation", { variation: "v2", model: "m2", edgeA: "ea", edgeB: "eb" }));
    state = { ...state, nodes: branched.graph.nodes, edges: branched.graph.edges };
    state = withVariationText(state, "p-variation", "Create a glass facade");
    state = ok(editNodeGeometry(state, "v2", { op: "volume", id: "tower", field: "floorCount", value: 8 }, { artifact: "x", revision: "y" }, NOW)).state;
    return commitVariations(state, newId, NOW);
  };

  it("keeps both branches distinct while the source stays untouched", () => {
    const state = build();
    expect(spec(state, "p-model").facade.style).toBe("grid"); // glass follow-up
    expect(spec(state, "m2").volumes[1].floorCount).toBe(8);
    expect(spec(state, "p-model")).not.toEqual(spec(state, "m2"));
    expect(state.artifacts.a1.metadata.brief).toEqual({ prompt: "A terraced stepped tower with a podium", refinement: "" });
    expect(spec(state, "p-generation").volumes[1].floorCount).not.toBe(8);
  });
  it("records an artifact and revision per branch pointing at the shared parent", () => {
    const state = build();
    const branchArtifacts = ["p-variation", "v2"].map((id) => state.nodes.find((x) => x.id === id)!.artifactId!);
    expect(branchArtifacts.every(Boolean)).toBe(true);
    const revs = Object.values(state.revisions).filter((r) => branchArtifacts.includes(r.childArtifactId));
    expect(revs).toHaveLength(2);
    revs.forEach((r) => expect(r.parentArtifactId).toBe("a1"));
    expect(revs.map((r) => r.change).sort()).toEqual(["parameters", "prompt"]);
    expect(lineageOf(state.revisions, state.artifacts, branchArtifacts[1]).map((e) => e.artifactId)).toEqual(["a1", branchArtifacts[1]]);
  });
  it("is idempotent and skips pass-through variations", () => {
    const state = build();
    expect(commitVariations(state, newId, NOW)).toEqual(state);
    const plain = generated();
    expect(commitVariations(plain, newId, NOW)).toEqual(plain);
  });
  it("snapshots again only when a recipe changes, keeping the earlier artifact", () => {
    const state = build();
    const before = state.nodes.find((x) => x.id === "p-variation")!.artifactId!;
    const changed = commitVariations(withVariationText(state, "p-variation", "Create a glass facade with a crown"), newId, NOW);
    const after = changed.nodes.find((x) => x.id === "p-variation")!.artifactId!;
    expect(after).not.toBe(before);
    expect(changed.artifacts[before]).toEqual(state.artifacts[before]);
    expect(versionsOf(changed, "p-variation", after).map((v) => v.current)).toEqual([false, true]);
  });
  it("produces a project that satisfies the persisted schema and survives a JSON round trip", () => {
    const state = build();
    const base = createWorkflowProject({ id: "p", name: "Branches", now: NOW, prompt: "x tower", refinement: "" });
    const project = { ...base, graph: { nodes: state.nodes, edges: state.edges }, artifacts: state.artifacts, jobs: state.jobs, revisions: state.revisions };
    const reloaded = siftProjectV2Schema.parse(JSON.parse(JSON.stringify(project)));
    const rs = { nodes: reloaded.graph.nodes, edges: reloaded.graph.edges, artifacts: reloaded.artifacts, jobs: reloaded.jobs, revisions: reloaded.revisions };
    expect(spec(rs, "m2")).toEqual(spec(state, "m2"));
    expect(spec(rs, "p-model")).toEqual(spec(state, "p-model"));
  });
});

describe("restoreVersion", () => {
  it("lists generation versions and points back to an earlier one without deleting later ones", () => {
    let state = generated();
    state = ok(editNodeGeometry(state, "p-generation", { op: "volume", id: "tower", field: "floorCount", value: 7 }, { artifact: "a2", revision: "r1" }, NOW)).state;
    expect(versionsOf(state, "p-generation", "a2").map((v) => v.artifactId)).toEqual(["a1", "a2"]);
    const back = ok(restoreVersion(state, "p-generation", "a1")).state;
    expect(back.nodes.find((x) => x.id === "p-generation")?.artifactId).toBe("a1");
    expect(Object.keys(back.artifacts)).toEqual(["a1", "a2"]);
    expect(spec(back, "p-generation").volumes[1].floorCount).not.toBe(7);
    expect(restoreVersion(state, "p-generation", "nope")).toMatchObject({ ok: false });
    expect(restoreVersion(state, "p-variation", "a1")).toMatchObject({ ok: false });
  });
});

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { IDBFactory } from "fake-indexeddb";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { siftProjectV2Schema, type SiftProjectV2 } from "./contracts";
import { createWorkflowProject, projectSignature } from "./projects";
import { renderInputKey, parseRenderSettings } from "./render-settings";
import { addNode, branchFrom, commitVariations, connectNodes, editNodeGeometry, evaluateGraph, recordRender, type RunInput } from "./workflow";

const NOW = "2026-10-09T10:00:00.000Z";
const ok = <T extends { ok: boolean }>(r: T) => { if (!r.ok) throw new Error(JSON.stringify(r)); return r as Extract<T, { ok: true }>; };
let n = 0;
const uid = (prefix: string) => `${prefix}-${(n += 1)}`;

/** The Terraced Tower board: prompt → generation → two branches (follow-up prompt / parameter edit) → models + a render. */
function buildTerracedTowerBoard(): SiftProjectV2 {
  const base = createWorkflowProject({ id: "tower", name: "Terraced Tower Study", now: NOW, prompt: "A terraced stepped office tower with a public podium", refinement: "" });
  let state: RunInput = { ...base.graph, artifacts: base.artifacts, jobs: base.jobs, revisions: base.revisions };
  const branched = ok(branchFrom(state, "tower-generation", { variation: "tower-vb", model: "tower-mb", edgeA: "eb1", edgeB: "eb2" }));
  state = { ...state, nodes: branched.graph.nodes, edges: branched.graph.edges };
  state = { ...state, nodes: state.nodes.map((node) => (node.id === "tower-variation" ? { ...node, params: { ...node.params, text: "glass facade, crown roof" } } : node)) };
  state = ok(editNodeGeometry(state, "tower-vb", { op: "volume", id: "tower", field: "floorCount", value: 9 }, { artifact: "x", revision: "y" }, NOW)).state;
  state = commitVariations(state, uid, NOW);
  state = { ...state, ...addNode(state, "render", "tower-render", { x: 1250, y: 120 }) };
  state = { ...state, ...ok(connectNodes(state, { source: "tower-variation", sourceHandle: "spec", target: "tower-render", targetHandle: "model" }, "er")).graph };
  state = { ...state, nodes: state.nodes.map((node) => (node.id === "tower-render" ? { ...node, params: { preset: "front", resolution: "1024x1024", background: "transparent" } } : node)) };
  state = ok(recordRender(state, "tower-render", { artifactId: "render-1", width: 1024, height: 1024, bytes: 4 }, NOW)).state;
  return siftProjectV2Schema.parse({
    ...base,
    viewport: { x: -140.5, y: 32, zoom: 0.62 },
    settings: { provider: "procedural", viewer: { preset: "top", mode: "clay", grid: false, axes: true, shadows: false } },
    graph: { nodes: state.nodes, edges: state.edges },
    artifacts: state.artifacts, jobs: state.jobs, revisions: state.revisions,
  });
}

const specOf = (s: RunInput, id: string) => { const r = evaluateGraph(s, s.artifacts)[id]; if (r.status === "blocked" || r.output.kind !== "spec") throw new Error("no spec for " + id); return r.output.spec; };
const asState = (p: SiftProjectV2): RunInput => ({ nodes: p.graph.nodes, edges: p.graph.edges, artifacts: p.artifacts, jobs: p.jobs, revisions: p.revisions });

describe("full board restore from IndexedDB", () => {
  beforeEach(() => { globalThis.indexedDB = new IDBFactory(); vi.resetModules(); });

  it("restores nodes, edges, viewport, settings, specs, revisions, jobs, both branches and the render asset after a reload", async () => {
    const original = buildTerracedTowerBoard();
    const bytes = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
    const storage = await import("./storage");
    await storage.saveAsset("asset:render-1", new Blob([bytes], { type: "image/png" }));
    await storage.saveProject(original);

    vi.resetModules(); // simulates a full page refresh: no in-memory state survives
    const reopened = await import("./storage");
    const [restored] = await reopened.listProjects();

    expect(restored).toEqual(original);
    expect(restored.viewport).toEqual({ x: -140.5, y: 32, zoom: 0.62 });
    expect(restored.settings.viewer).toEqual({ preset: "top", mode: "clay", grid: false, axes: true, shadows: false });
    expect(restored.graph.nodes.filter((nd) => nd.type === "variation").map((nd) => nd.params.label)).toEqual(["Branch A", "Branch B"]);
    expect(Object.keys(restored.revisions).length).toBeGreaterThanOrEqual(2);
    expect(Object.values(restored.jobs)[0]).toMatchObject({ status: "completed" });

    const before = asState(original);
    const after = asState(restored);
    ["tower-model", "tower-mb", "tower-render"].forEach((id) => expect(specOf(after, id)).toEqual(specOf(before, id)));
    expect(specOf(after, "tower-model")).not.toEqual(specOf(after, "tower-mb")); // both branches still differ
    expect(evaluateGraph(after, after.artifacts)["tower-render"].status).toBe("ready"); // render still matches model + settings

    const asset = await reopened.loadAsset(restored.artifacts["render-1"].storageKey);
    expect([...new Uint8Array(await asset!.arrayBuffer())]).toEqual([...bytes]);
    expect(restored.artifacts["render-1"].metadata.inputKey).toBe(renderInputKey(specOf(after, "tower-render"), parseRenderSettings(restored.graph.nodes.find((nd) => nd.id === "tower-render")!.params)));
  });

  it("opens projects saved before viewer settings existed", async () => {
    const original = buildTerracedTowerBoard();
    const { viewer: _viewer, ...settings } = original.settings;
    void _viewer;
    const storage = await import("./storage");
    await storage.saveProject({ ...original, settings });
    expect((await storage.listProjects())[0].settings).toEqual({ provider: "procedural" });
  });

  it("keeps independent projects and their assets separate", async () => {
    const storage = await import("./storage");
    await storage.saveProject(buildTerracedTowerBoard());
    await storage.saveProject({ ...createWorkflowProject({ id: "other", name: "Other", now: NOW, prompt: "Twin towers", refinement: "" }) });
    expect((await storage.listProjects()).map((p) => p.id).sort()).toEqual(["other", "tower"]);
  });
});

describe("projectSignature", () => {
  const p = buildTerracedTowerBoard();
  const sig = (x: SiftProjectV2) => projectSignature(x);
  it("is stable across key order, timestamps and unrelated rewrites", () => {
    const shuffled = { ...p, updatedAt: "2030-01-01T00:00:00.000Z", graph: { nodes: p.graph.nodes.map((nd) => ({ params: nd.params, position: nd.position, type: nd.type, id: nd.id, ...(nd.artifactId ? { artifactId: nd.artifactId } : {}) })), edges: p.graph.edges } };
    expect(sig(shuffled as SiftProjectV2)).toBe(sig(p));
  });
  it("changes for every persisted kind of edit", () => {
    const base = sig(p);
    const moved = { ...p, graph: { ...p.graph, nodes: p.graph.nodes.map((nd, i) => (i === 0 ? { ...nd, position: { x: nd.position.x + 5, y: nd.position.y } } : nd)) } };
    const typed = { ...p, graph: { ...p.graph, nodes: p.graph.nodes.map((nd) => (nd.type === "prompt" ? { ...nd, params: { text: "Different brief" } } : nd)) } };
    const panned = { ...p, viewport: { ...p.viewport, x: p.viewport.x - 20 } };
    const viewer = { ...p, settings: { ...p.settings, viewer: { ...p.settings.viewer!, mode: "wireframe" as const } } };
    const renamed = { ...p, name: "Renamed" };
    const rewired = { ...p, graph: { ...p.graph, edges: p.graph.edges.slice(1) } };
    [moved, typed, panned, viewer, renamed, rewired].forEach((changed) => expect(sig(changed)).not.toBe(base));
    expect(sig({ ...p, viewport: { ...p.viewport, x: p.viewport.x + 0.001 } })).toBe(base); // sub-pixel pan noise is ignored
  });
});

describe("no localStorage blobs", () => {
  const walk = (dir: string): string[] => readdirSync(dir).flatMap((name) => { const full = join(dir, name); return statSync(full).isDirectory() ? (name === "__pycache__" ? [] : walk(full)) : /\.(ts|tsx)$/.test(name) ? [full] : []; });
  it("application code never touches localStorage or sessionStorage", () => {
    const offenders = walk("src").filter((file) => !file.endsWith("persistence.test.ts") && /\b(localStorage|sessionStorage)\b/.test(readFileSync(file, "utf8")));
    expect(offenders).toEqual([]);
  });
});

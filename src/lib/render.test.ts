import { IDBFactory } from "fake-indexeddb";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_RENDER_SETTINGS, RESOLUTIONS, describeRender, parseRenderSettings, renderInputKey, supportedResolutions } from "./render-settings";
import { workflowGraph } from "./projects";
import { deriveBuildingSpec } from "./typologies";
import { evaluateGraph, recordRender, runGeneration, type RunInput } from "./workflow";

const NOW = "2026-10-09T10:00:00.000Z";
const ok = <T extends { ok: boolean }>(r: T) => { if (!r.ok) throw new Error(JSON.stringify(r)); return r as Extract<T, { ok: true }>; };
const withRender = (): RunInput => {
  const g = workflowGraph("p", "A terraced stepped tower with a podium", "");
  const graph = { nodes: [...g.nodes, { id: "r", type: "render" as const, position: { x: 0, y: 0 }, params: {} }], edges: [...g.edges, { id: "er", source: "p-variation", sourcePort: "spec", target: "r", targetPort: "model" }] };
  return ok(runGeneration({ ...graph, artifacts: {}, jobs: {}, revisions: {} }, "p-generation", "procedural", { artifact: "a1", job: "j1", revision: "r0" }, NOW)).state;
};
const rec = (id: string) => ({ artifactId: id, width: 1600, height: 900, bytes: 1234 });

describe("supportedResolutions", () => {
  const big = { maxRenderbufferSize: 8192, maxViewportDims: [8192, 8192] as [number, number] };
  it("offers all sizes on a capable GPU", () => expect(supportedResolutions(big)).toEqual(["1024x1024", "1600x900", "1920x1080"]));
  it("offers nothing when WebGL is unavailable", () => expect(supportedResolutions(null)).toEqual([]));
  it("withholds sizes the GPU cannot allocate", () => {
    expect(supportedResolutions({ maxRenderbufferSize: 1600, maxViewportDims: [1600, 1600] })).toEqual(["1024x1024", "1600x900"]);
    expect(supportedResolutions({ maxRenderbufferSize: 1024, maxViewportDims: [1024, 1024] })).toEqual(["1024x1024"]);
    expect(supportedResolutions({ maxRenderbufferSize: 4096, maxViewportDims: [4096, 800] })).toEqual([]);
  });
  it("withholds full HD on low-memory devices", () => {
    expect(supportedResolutions({ ...big, deviceMemoryGb: 2 })).toEqual(["1024x1024", "1600x900"]);
    expect(supportedResolutions({ ...big, deviceMemoryGb: 8 })).toContain("1920x1080");
  });
});

describe("parseRenderSettings / renderInputKey", () => {
  it("falls back to defaults for missing or invalid values", () => {
    expect(parseRenderSettings({})).toEqual(DEFAULT_RENDER_SETTINGS);
    expect(parseRenderSettings({ preset: "bird", mode: 3, resolution: "4k", lighting: "dusk" })).toEqual({ ...DEFAULT_RENDER_SETTINGS, lighting: "dusk" });
  });
  it("keys change with the model or any setting, and are stable otherwise", () => {
    const spec = deriveBuildingSpec("A tower");
    const base = renderInputKey(spec, DEFAULT_RENDER_SETTINGS);
    expect(renderInputKey(deriveBuildingSpec("A tower"), { ...DEFAULT_RENDER_SETTINGS })).toBe(base);
    expect(renderInputKey(spec, { ...DEFAULT_RENDER_SETTINGS, preset: "top" })).not.toBe(base);
    expect(renderInputKey(spec, { ...DEFAULT_RENDER_SETTINGS, resolution: "1024x1024" })).not.toBe(base);
    expect(renderInputKey(deriveBuildingSpec("Twin towers"), DEFAULT_RENDER_SETTINGS)).not.toBe(base);
  });
  it("exposes exactly the required sizes", () => {
    expect(Object.values(RESOLUTIONS).map((r) => [r.width, r.height])).toEqual([[1024, 1024], [1600, 900], [1920, 1080]]);
    expect(describeRender(DEFAULT_RENDER_SETTINGS)).toBe("1600×900 · axonometric · shaded");
  });
});

describe("render nodes in the workflow", () => {
  it("needs a render until recorded, then is ready", () => {
    const state = withRender();
    expect(evaluateGraph(state, state.artifacts).r).toMatchObject({ status: "pending", message: expect.stringMatching(/Not rendered/) });
    const next = ok(recordRender(state, "r", rec("img1"), NOW)).state;
    expect(evaluateGraph(next, next.artifacts).r.status).toBe("ready");
    expect(next.artifacts.img1).toMatchObject({ kind: "render-png", sourceNodeId: "r", storageKey: "asset:img1", metadata: { width: 1600, height: 900, parentArtifactId: "a1" } });
  });
  it("goes out of date when settings or the upstream model change, and keeps the old render", () => {
    const rendered = ok(recordRender(withRender(), "r", rec("img1"), NOW)).state;
    const changedSettings = { ...rendered, nodes: rendered.nodes.map((n) => (n.id === "r" ? { ...n, params: { preset: "top" } } : n)) };
    expect(evaluateGraph(changedSettings, changedSettings.artifacts).r).toMatchObject({ status: "pending", message: expect.stringMatching(/changed/) });
    const edited = { ...rendered, nodes: rendered.nodes.map((n) => (n.id === "p-variation" ? { ...n, params: { text: "glass" } } : n)) };
    expect(evaluateGraph(edited, edited.artifacts).r.status).toBe("pending");
    const again = ok(recordRender(changedSettings, "r", rec("img2"), NOW)).state;
    expect(Object.keys(again.artifacts)).toEqual(expect.arrayContaining(["img1", "img2"]));
    expect(evaluateGraph(again, again.artifacts).r.status).toBe("ready");
  });
  it("records the settings used and refuses unconnected or non-render nodes", () => {
    const state = withRender();
    const withSettings = { ...state, nodes: state.nodes.map((n) => (n.id === "r" ? { ...n, params: { preset: "front", resolution: "1024x1024" } } : n)) };
    expect(ok(recordRender(withSettings, "r", { ...rec("img"), width: 1024, height: 1024 }, NOW)).state.artifacts.img.metadata.settings).toMatchObject({ preset: "front", resolution: "1024x1024" });
    expect(recordRender(state, "p-model", rec("x"), NOW)).toMatchObject({ ok: false });
    const lonely = { ...state, edges: state.edges.filter((e) => e.id !== "er") };
    expect(recordRender(lonely, "r", rec("x"), NOW)).toMatchObject({ ok: false });
  });
});

describe("asset storage (fake IndexedDB)", () => {
  beforeEach(() => { globalThis.indexedDB = new IDBFactory(); vi.resetModules(); });
  it("round-trips binary assets and reports missing ones", async () => {
    const storage = await import("./storage");
    await storage.saveAsset("asset:1", new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" }));
    const blob = await storage.loadAsset("asset:1");
    expect(blob?.type).toBe("image/png");
    expect([...new Uint8Array(await blob!.arrayBuffer())]).toEqual([137, 80, 78, 71]);
    expect(await storage.loadAsset("asset:none")).toBeNull();
    await storage.deleteAssets(["asset:1"]);
    expect(await storage.loadAsset("asset:1")).toBeNull();
  });
  it("deleting a project removes its render assets", async () => {
    const storage = await import("./storage");
    const { createWorkflowProject } = await import("./projects");
    const base = createWorkflowProject({ id: "proj", name: "P", now: NOW, prompt: "A tower", refinement: "" });
    const art = { id: "img", kind: "render-png" as const, sourceNodeId: base.graph.nodes[3].id, createdAt: NOW, storageKey: "asset:img", metadata: {} };
    await storage.saveAsset("asset:img", new Blob(["x"], { type: "image/png" }));
    await storage.saveProject({ ...base, artifacts: { ...base.artifacts, img: art } });
    await storage.deleteProject("proj");
    expect(await storage.loadAsset("asset:img")).toBeNull();
  });
});

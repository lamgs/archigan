import { describe, expect, it } from "vitest";
import { buildingSpecSchema } from "./contracts";
import { computeLayout } from "./geometry";
import { workflowGraph } from "./projects";
import { applyEdit, applyEdits, mergeEdit, type SpecEdit } from "./spec-edit";
import { deriveBuildingSpec } from "./typologies";
import { editNodeGeometry, evaluateGraph, runGeneration, variationEdits, type RunInput } from "./workflow";

const NOW = "2026-10-09T10:00:00.000Z";
const terraced = () => deriveBuildingSpec("A terraced stepped office tower with a public podium");
const ok = <T extends { ok: boolean }>(r: T) => { if (!r.ok) throw new Error(JSON.stringify(r)); return r as Extract<T, { ok: true }>; };

describe("applyEdit", () => {
  it("changes only the targeted field and does not mutate the input", () => {
    const spec = terraced();
    const before = structuredClone(spec);
    const { spec: next } = ok(applyEdit(spec, { op: "volume", id: "tower", field: "floorCount", value: 12 }));
    expect(spec).toEqual(before);
    expect(next.volumes.find((v) => v.id === "tower")?.floorCount).toBe(12);
    expect(next.volumes.find((v) => v.id === "podium")).toEqual(before.volumes.find((v) => v.id === "podium"));
  });
  it("makes parameter changes visible in the layout", () => {
    const spec = terraced();
    const tall = ok(applyEdit(spec, { op: "volume", id: "tower", field: "floorCount", value: 40 })).spec;
    expect(computeLayout(tall).bounds.max[1]).toBeGreaterThan(computeLayout(spec).bounds.max[1]);
    const twisted = ok(applyEdit(spec, { op: "volume", id: "tower", field: "rotationDegrees", value: 60 })).spec;
    expect(computeLayout(twisted).slabs.at(-2)?.rotationY).not.toBe(0);
  });
  it("rejects out-of-range values, unknown ids and impossible combinations with a message", () => {
    const spec = terraced();
    expect(applyEdit(spec, { op: "floorHeight", value: 99 })).toMatchObject({ ok: false });
    expect(applyEdit(spec, { op: "volume", id: "nope", field: "taper", value: 0.1 })).toMatchObject({ ok: false, message: expect.stringMatching(/Unknown volume/) });
    expect(applyEdit(spec, { op: "volume", id: "tower", field: "startFloor", value: 119 })).toMatchObject({ ok: false });
    expect(applyEdit(spec, { op: "material", id: "ghost", color: "#000000" })).toMatchObject({ ok: false });
    expect(applyEdit(spec, { op: "material", id: "accent", color: "red" })).toMatchObject({ ok: false });
  });
  it("switches footprint shape both ways and adds/removes setbacks", () => {
    const circle = ok(applyEdit(terraced(), { op: "footprintType", type: "circle" })).spec;
    expect(circle.footprint.type).toBe("circle");
    expect(ok(applyEdit(circle, { op: "footprintType", type: "rectangle" })).spec.footprint.type).toBe("rectangle");
    const flat = ok(applyEdit(terraced(), { op: "setback", id: "tower", every: null })).spec;
    expect(flat.volumes[1].setbackEvery).toBeUndefined();
    expect(buildingSpecSchema.safeParse(flat).success).toBe(true);
    const stepped = ok(applyEdit(flat, { op: "setback", id: "tower", every: 5, amount: 2 })).spec;
    expect(stepped.volumes[1]).toMatchObject({ setbackEvery: 5, setbackAmount: 2 });
  });
});

describe("edit lists", () => {
  it("keeps only the latest edit per control and replays in order", () => {
    const a: SpecEdit = { op: "volume", id: "tower", field: "taper", value: 0.1 };
    const b: SpecEdit = { op: "volume", id: "tower", field: "taper", value: 0.3 };
    const c: SpecEdit = { op: "roof", style: "crown" };
    const merged = mergeEdit(mergeEdit(mergeEdit([], a), c), b);
    expect(merged).toEqual([c, b]);
    const spec = applyEdits(terraced(), merged);
    expect(spec.volumes[1].taper).toBe(0.3);
    expect(spec.roof.style).toBe("crown");
  });
  it("skips edits that no longer apply instead of throwing", () => {
    expect(applyEdits(deriveBuildingSpec("A tower with a spiral twist"), [{ op: "volume", id: "tower-b", field: "taper", value: 0.2 }])).toBeDefined();
  });
});

describe("editNodeGeometry (revisions, not overwrites)", () => {
  const base = (): RunInput => {
    const run = ok(runGeneration({ ...workflowGraph("p", "A terraced stepped tower with a podium", ""), artifacts: {}, jobs: {}, revisions: {} }, "p-generation", "procedural", { artifact: "a1", job: "j1", revision: "r0" }, NOW));
    return run.state;
  };
  const edit: SpecEdit = { op: "volume", id: "tower", field: "floorCount", value: 9 };

  it("on a generation node creates a child artifact and revision, keeping the source", () => {
    const state = base();
    const source = structuredClone(state.artifacts.a1);
    const next = ok(editNodeGeometry(state, "p-generation", edit, { artifact: "a2", revision: "r1" }, NOW)).state;
    expect(next.artifacts.a1).toEqual(source);
    expect(next.artifacts.a2.metadata).toMatchObject({ origin: "parameters", parentArtifactId: "a1" });
    expect(next.revisions.r1).toMatchObject({ parentArtifactId: "a1", childArtifactId: "a2", change: "parameters" });
    expect(next.nodes.find((n) => n.id === "p-generation")?.artifactId).toBe("a2");
    const out = evaluateGraph(next, next.artifacts)["p-model"];
    if (out.status === "blocked" || out.output.kind !== "spec") throw new Error("expected spec");
    expect(out.output.spec.volumes[1].floorCount).toBe(9);
    expect(state.nodes.find((n) => n.id === "p-generation")?.artifactId).toBe("a1"); // input state untouched
  });
  it("on a variation node stores the edit on the node and leaves artifacts untouched", () => {
    const state = base();
    const next = ok(editNodeGeometry(state, "p-variation", edit, { artifact: "a2", revision: "r1" }, NOW)).state;
    expect(next.artifacts).toEqual(state.artifacts);
    expect(variationEdits(next.nodes.find((n) => n.id === "p-variation")!)).toEqual([edit]);
    const gen = evaluateGraph(next, next.artifacts)["p-generation"];
    const mod = evaluateGraph(next, next.artifacts)["p-model"];
    if (gen.status === "blocked" || mod.status === "blocked" || gen.output.kind !== "spec" || mod.output.kind !== "spec") throw new Error("expected spec");
    expect(gen.output.spec.volumes[1].floorCount).not.toBe(9);
    expect(mod.output.spec.volumes[1].floorCount).toBe(9);
  });
  it("refuses invalid edits, un-run nodes, and non-editable nodes without changing state", () => {
    const state = base();
    expect(editNodeGeometry(state, "p-generation", { op: "floorHeight", value: 99 }, { artifact: "x", revision: "y" }, NOW)).toMatchObject({ ok: false });
    expect(editNodeGeometry(state, "p-model", edit, { artifact: "x", revision: "y" }, NOW)).toMatchObject({ ok: false });
    const fresh: RunInput = { ...workflowGraph("q", "A tower", ""), artifacts: {}, jobs: {}, revisions: {} };
    expect(editNodeGeometry(fresh, "q-generation", edit, { artifact: "x", revision: "y" }, NOW)).toMatchObject({ ok: false, message: expect.stringMatching(/no model/) });
  });
});

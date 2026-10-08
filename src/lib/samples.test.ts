import { describe, expect, it } from "vitest";
import { siftProjectV2Schema } from "./contracts";
import { computeLayout } from "./geometry";
import { layoutComplexity } from "./limits";
import { copyFromSample } from "./projects";
import { buildTerracedTowerStudy, FEATURED_SAMPLE_ID, SAMPLE_BLURBS, sampleProjects } from "./samples";
import { evaluateGraph } from "./workflow";

const featured = () => sampleProjects.find((s) => s.id === FEATURED_SAMPLE_ID)!;
const specOf = (p: ReturnType<typeof featured>, nodeId: string) => {
  const r = evaluateGraph(p.graph, p.artifacts)[nodeId];
  if (r.status === "blocked" || r.output.kind !== "spec") throw new Error(`no spec for ${nodeId}`);
  return r.output.spec;
};

describe("bundled samples", () => {
  it("are valid, uniquely identified, within mesh limits, and described", () => {
    expect(new Set(sampleProjects.map((s) => s.id)).size).toBe(sampleProjects.length);
    sampleProjects.forEach((s) => {
      expect(siftProjectV2Schema.safeParse(s).success, s.name).toBe(true);
      expect(SAMPLE_BLURBS[s.id], s.id).toBeTruthy();
      const spec = specOf(s, s.graph.nodes.find((n) => n.type === "model")!.id);
      expect(layoutComplexity(computeLayout(spec)).ok, s.name).toBe(true);
    });
    expect(sampleProjects[0].id).toBe(FEATURED_SAMPLE_ID);
  });
  it("are deterministic", () => expect(buildTerracedTowerStudy()).toEqual(buildTerracedTowerStudy()));

  it("covers three structurally distinct typologies", () => {
    const byId = (id: string) => specOf(sampleProjects.find((s) => s.id === id)!, `${id}-model`);
    const terraced = specOf(featured(), "sample-terraced-tower-model-b");
    const twin = byId("sample-twin-towers");
    const cylinder = byId("sample-cylinder");
    expect(terraced.volumes.some((v) => v.setbackEvery !== undefined)).toBe(true);
    expect(twin.volumes.filter((v) => v.role === "tower")).toHaveLength(2);
    expect(cylinder.footprint.type).toBe("circle");
    expect(terraced.footprint.type).toBe("rectangle");
  });
});

describe("Terraced Tower Study", () => {
  it("contains two visible, labelled branches from one generation with recorded lineage", () => {
    const p = featured();
    const variations = p.graph.nodes.filter((n) => n.type === "variation");
    expect(variations.map((n) => n.params.label).sort()).toEqual(["Branch A", "Branch B"]);
    const generation = p.graph.nodes.find((n) => n.type === "generation")!;
    expect(p.graph.edges.filter((e) => e.source === generation.id)).toHaveLength(2);
    const parent = generation.artifactId!;
    const revisions = Object.values(p.revisions).filter((r) => r.parentArtifactId === parent);
    expect(revisions.map((r) => r.change).sort()).toEqual(["parameters", "prompt"]);
    variations.forEach((v) => expect(p.artifacts[v.artifactId!]).toBeTruthy());
  });
  it("branches differ in real geometry and the source is untouched", () => {
    const p = featured();
    const a = specOf(p, "sample-terraced-tower-model");
    const b = specOf(p, "sample-terraced-tower-model-b");
    expect(a.facade.style).toBe("grid");
    expect(Object.values(a.materials).some((m) => m.kind === "glass")).toBe(true);
    expect(b.volumes.find((v) => v.id === "tower")).toMatchObject({ floorCount: 16, setbackEvery: 3, setbackAmount: 2.2 });
    expect(Object.values(specOf(p, "sample-terraced-tower-generation").materials).some((m) => m.kind === "glass")).toBe(false);
  });
  it("includes a Render node that renders itself on open, not a stale image", () => {
    const p = featured();
    const render = p.graph.nodes.find((n) => n.type === "render")!;
    expect(render.params).toMatchObject({ autoRender: true, resolution: "1600x900" });
    expect(render.artifactId).toBeUndefined();
    expect(Object.values(p.artifacts).some((a) => a.kind === "render-png")).toBe(false);
    expect(evaluateGraph(p.graph, p.artifacts)[render.id].status).toBe("pending");
  });
});

describe("copyFromSample", () => {
  it("makes an independent, uniquely named copy and keeps the whole board", () => {
    const sample = featured();
    const before = structuredClone(sample);
    const copy = copyFromSample(sample, "new-project", "2026-10-09T00:00:00.000Z", [sample.name]);
    expect(copy).toMatchObject({ id: "new-project", name: "Terraced Tower Study 2" });
    expect(copy.graph).toEqual(sample.graph);
    expect(Object.keys(copy.revisions)).toEqual(Object.keys(sample.revisions));
    copy.graph.nodes[0].params = { text: "changed" };
    expect(sample).toEqual(before); // the bundled sample never changes
    expect(siftProjectV2Schema.safeParse(copy).success).toBe(true);
  });
});

describe("sample wording is honoured by the local interpreter", () => {
  it("every word we promise in the help text changes the result", async () => {
    const { deriveBuildingSpec, detectTypology, INTERPRETER_HELP } = await import("./typologies");
    expect(INTERPRETER_HELP).toMatch(/twin/);
    const base = deriveBuildingSpec("A building");
    expect(detectTypology("twin towers")).toBe("twin");
    expect(detectTypology("a round tower")).toBe("cylindrical");
    expect(detectTypology("a spiral tower")).toBe("rotated");
    expect(detectTypology("a low-rise pavilion")).toBe("low-rise");
    expect(deriveBuildingSpec("A building glass").facade.style).not.toBe(base.facade.style);
    expect(deriveBuildingSpec("A building terracotta").materials).not.toEqual(base.materials);
    expect(deriveBuildingSpec("A building tower").volumes[1].floorCount).not.toBe(base.volumes[1].floorCount);
    expect(deriveBuildingSpec("A building garden").roof.style).toBe("terrace");
    expect(deriveBuildingSpec("A building museum").floorHeight).toBeGreaterThan(base.floorHeight);
  });
});

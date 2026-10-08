import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { buildingSpecSchema } from "./contracts";
import { backupFilename, BACKUP_FORMAT, exportProjectJson, parseProjectJson } from "./backup";
import { computeLayout } from "./geometry";
import { countObjectTriangles, layoutComplexity, MESH_LIMITS } from "./limits";
import { legacySamples } from "./legacy-fixtures";
import { createWorkflowProject } from "./projects";
import { applyEdit } from "./spec-edit";
import { buildBuildingGroup, disposeBuildingGroup } from "./three-building";
import { deriveBuildingSpec } from "./typologies";

const NOW = "2026-10-09T10:00:00.000Z";
const project = () => createWorkflowProject({ id: "p1", name: "Terraced Tower: 1/2?", now: NOW, prompt: "A terraced stepped tower", refinement: "" });

describe("mesh limits", () => {
  it("accepts every typology", () => {
    ["A terraced stepped tower", "Twin towers", "A cylindrical glass tower", "A tower with a spiral twist", "A low pavilion"].forEach((brief) => {
      const c = layoutComplexity(computeLayout(deriveBuildingSpec(brief)));
      expect(c.ok).toBe(true);
      expect(c.triangles).toBeGreaterThan(100);
    });
  });
  it("counts exactly what the builder creates", () => {
    const spec = deriveBuildingSpec("A cylindrical glass residential tower");
    const layout = computeLayout(spec);
    const group = buildBuildingGroup(spec, layout);
    const built = countObjectTriangles(group) - 12; // minus the ground plate (a box)
    expect(layoutComplexity(layout).triangles).toBe(built);
    disposeBuildingGroup(group);
  });
  it("rejects edits that blow the budget with an explanatory message and leaves the spec untouched", () => {
    const circle = deriveBuildingSpec("A cylindrical glass tower");
    const base = buildingSpecSchema.parse({ ...circle, volumes: Array.from({ length: 10 }, (_, i) => ({ id: `v${i}`, role: "wing", startFloor: 0, floorCount: 9, footprintScale: 0.5, offsetX: 0, offsetZ: 0, rotationDegrees: 0, taper: 0, materialId: "glass" })) });
    const before = layoutComplexity(computeLayout(base));
    expect(before.ok).toBe(true);
    const heavy = applyEdit(base, { op: "volume", id: "v0", field: "floorCount", value: 120 });
    expect(heavy).toMatchObject({ ok: false, message: expect.stringMatching(/triangles/) });
    expect(applyEdit(base, { op: "volume", id: "v0", field: "floorCount", value: 20 }).ok).toBe(true);
    expect(before.triangles).toBeLessThan(MESH_LIMITS.maxTriangles);
  });
});

describe("GPU resource cleanup", () => {
  it("disposes every geometry and every distinct material exactly once", () => {
    ["shaded", "clay", "glass-concrete", "wireframe"].forEach((mode) => {
      const spec = deriveBuildingSpec("Twin towers rising from a shared podium");
      const group = buildBuildingGroup(spec, undefined, mode as "shaded");
      const geometries = new Set<THREE.BufferGeometry>();
      const materials = new Set<THREE.Material>();
      let geometryDisposals = 0;
      let materialDisposals = 0;
      group.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        geometries.add(child.geometry);
        (Array.isArray(child.material) ? child.material : [child.material]).forEach((m) => materials.add(m));
      });
      geometries.forEach((g) => g.addEventListener("dispose", () => { geometryDisposals += 1; }));
      materials.forEach((m) => m.addEventListener("dispose", () => { materialDisposals += 1; }));
      disposeBuildingGroup(group);
      expect(geometries.size).toBeGreaterThan(10);
      expect(geometryDisposals).toBe(geometries.size);
      expect(materialDisposals).toBe(materials.size);
    });
  });
});

describe("project backup and import", () => {
  it("round-trips a project through a backup file", () => {
    const p = project();
    const result = parseProjectJson(exportProjectJson(p), () => "new-id");
    expect(result).toMatchObject({ ok: true, warnings: [] });
    if (result.ok) expect(result.project).toEqual(p);
    expect(JSON.parse(exportProjectJson(p)).format).toBe(BACKUP_FORMAT);
  });
  it("accepts bare v2 records and legacy v1 projects", () => {
    expect(parseProjectJson(JSON.stringify(project()), () => "x").ok).toBe(true);
    const legacy = parseProjectJson(JSON.stringify(legacySamples[0]), () => "x");
    expect(legacy.ok).toBe(true);
    if (legacy.ok) expect(legacy.project.schemaVersion).toBe(2);
  });
  it("saves id collisions as copies and warns about files that are not included", () => {
    const p = project();
    const withRender = { ...p, artifacts: { ...p.artifacts, r: { id: "r", kind: "render-png" as const, sourceNodeId: "p1-model", createdAt: NOW, storageKey: "asset:r", metadata: {} } } };
    const result = parseProjectJson(exportProjectJson(withRender), () => "fresh", ["p1"]);
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.project.id).toBe("fresh");
      expect(result.warnings.join(" ")).toMatch(/already exists/);
      expect(result.warnings.join(" ")).toMatch(/1 image\/model file/);
    }
  });
  it("rejects garbage without throwing", () => {
    ["", "not json", "[]", "{}", JSON.stringify({ schemaVersion: 2 }), JSON.stringify({ format: BACKUP_FORMAT, project: { schemaVersion: 9 } })].forEach((text) => expect(parseProjectJson(text, () => "x")).toMatchObject({ ok: false }));
    expect(parseProjectJson("x".repeat(20_000_001), () => "x")).toMatchObject({ ok: false, error: expect.stringMatching(/too large/) });
  });
  it("rejects structurally broken projects (dangling references)", () => {
    const p = project();
    const broken = { ...p, graph: { ...p.graph, edges: [{ id: "e", source: "ghost", sourcePort: "spec", target: "p1-model", targetPort: "spec" }] } };
    expect(parseProjectJson(JSON.stringify(broken), () => "x")).toMatchObject({ ok: false });
  });
  it("makes safe filenames", () => {
    expect(backupFilename({ name: "Terraced Tower: 1/2?" })).toBe("terraced-tower-1-2.sift.json");
    expect(backupFilename({ name: "???" })).toBe("sift-project.sift.json");
  });
});

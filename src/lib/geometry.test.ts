import { describe, expect, it } from "vitest";
import { buildingSpecSchema, type BuildingSpec } from "./contracts";
import { computeLayout } from "./geometry";
import { deriveBuildingSpec, detectTypology } from "./typologies";

const base = (over: Partial<BuildingSpec["volumes"][number]> = {}, footprint: BuildingSpec["footprint"] = { type: "rectangle", width: 30, depth: 20 }): BuildingSpec =>
  buildingSpecSchema.parse({
    schemaVersion: 1, name: "t", units: "m", floorHeight: 4, footprint,
    volumes: [{ id: "v", role: "tower", startFloor: 0, floorCount: 10, footprintScale: 1, offsetX: 0, offsetZ: 0, rotationDegrees: 0, taper: 0, materialId: "m", ...over }],
    facade: { style: "horizontal", glazingRatio: 0.3 }, roof: { style: "flat" }, materials: { m: { kind: "concrete", color: "#999999" } },
  });
const floors = (spec: BuildingSpec) => computeLayout(spec).slabs.filter((s) => s.kind === "floor");
const width = (s: ReturnType<typeof floors>[number]) => (s.shape.type === "rectangle" ? s.shape.width : s.shape.radius * 2);

describe("computeLayout", () => {
  it("stacks one slab per floor at floor-height intervals", () => {
    const slabs = floors(base());
    expect(slabs).toHaveLength(10);
    expect(slabs.map((s) => s.y)).toEqual(Array.from({ length: 10 }, (_, i) => i * 4));
    expect(computeLayout(base()).bounds.max[1]).toBeCloseTo(9 * 4 + 4 * 0.92);
  });
  it("tapers linearly toward the top", () => {
    const slabs = floors(base({ taper: 0.5 }));
    expect(width(slabs[0])).toBeCloseTo(30);
    expect(width(slabs[9])).toBeCloseTo(15);
  });
  it("steps back every N floors", () => {
    const slabs = floors(base({ setbackEvery: 3, setbackAmount: 1 }));
    expect([0, 2, 3, 6, 9].map((i) => width(slabs[i]))).toEqual([30, 30, 28, 26, 24]);
  });
  it("applies total twist, offsets, and circular footprints", () => {
    const slabs = floors(base({ rotationDegrees: 90, offsetX: 5, offsetZ: -3 }, { type: "circle", radius: 10 }));
    expect(slabs[0].rotationY).toBe(0);
    expect(slabs[9].rotationY).toBeCloseTo(Math.PI / 2);
    expect(slabs[4]).toMatchObject({ x: 5, z: -3, shape: { type: "circle", radius: 10 } });
  });
  it("clamps and warns when setbacks would invert a volume", () => {
    const layout = computeLayout(base({ setbackEvery: 1, setbackAmount: 10 }));
    expect(layout.warnings[0]).toMatch(/clamped/);
    layout.slabs.forEach((s) => s.shape.type === "rectangle" && expect(s.shape.width).toBeGreaterThanOrEqual(2));
  });
  it("adds roof elements only for terrace and crown roofs, and no glazing for solid facades", () => {
    expect(computeLayout(base()).slabs.some((s) => s.kind === "roof")).toBe(false);
    expect(computeLayout({ ...base(), roof: { style: "crown" } }).slabs.at(-1)?.kind).toBe("roof");
    expect(floors({ ...base(), facade: { style: "solid", glazingRatio: 0.9 } }).every((s) => s.glazing === 0)).toBe(true);
  });
  it("bounds reflect rotation", () => {
    const flat = computeLayout(base({ floorCount: 1 }));
    const turned = computeLayout(base({ floorCount: 2, rotationDegrees: 90 }));
    expect(turned.bounds.max[0]).toBeGreaterThan(flat.bounds.max[0] - 1e-9);
  });
});

describe("typologies", () => {
  const briefs = {
    terraced: "A terraced stepped office tower with a public podium",
    twin: "Twin towers rising from a shared podium",
    cylindrical: "A cylindrical glass residential tower",
    rotated: "A slender tower with a spiral twist",
  } as const;
  it.each(Object.entries(briefs))("detects %s", (type, brief) => expect(detectTypology(brief.toLowerCase())).toBe(type));
  it("produces valid, structurally distinct specs", () => {
    const [terraced, twin, cyl, rot] = Object.values(briefs).map((b) => deriveBuildingSpec(b));
    expect(terraced.volumes.map((v) => v.role)).toEqual(["podium", "tower"]);
    expect(terraced.volumes[1].setbackEvery).toBeDefined();
    const towers = twin.volumes.filter((v) => v.role === "tower");
    expect(towers).toHaveLength(2);
    expect(towers[0].offsetX).toBeLessThan(0);
    expect(towers[1].offsetX).toBeGreaterThan(0);
    expect(cyl.footprint.type).toBe("circle");
    expect(rot.volumes[0].rotationDegrees).not.toBe(0);
    [terraced, twin, cyl, rot].forEach((s) => expect(computeLayout(s).slabs.length).toBeGreaterThan(10));
    // Distinct silhouettes: footprint extents differ by typology.
    const extents = [terraced, twin, cyl, rot].map((s) => computeLayout(s).bounds.max[0] - computeLayout(s).bounds.min[0]);
    expect(new Set(extents.map((e) => Math.round(e))).size).toBe(4);
  });
  it("is deterministic and reacts to parameter-bearing refinements", () => {
    expect({ ...deriveBuildingSpec("A tower", "glass"), name: "" }).toEqual({ ...deriveBuildingSpec("  A  TOWER ", "glass"), name: "" });
    expect(deriveBuildingSpec("A tower", "glass").facade.style).toBe("grid");
    expect(deriveBuildingSpec("A tower", "terracotta").facade.style).toBe("horizontal");
  });
  it("derives a valid spec for every bundled sample brief", async () => {
    const { legacySamples: sampleProjects } = await import("./legacy-fixtures");
    sampleProjects.forEach((s) => expect(buildingSpecSchema.safeParse(deriveBuildingSpec(s.prompt, s.refinement)).success).toBe(true));
  });
});

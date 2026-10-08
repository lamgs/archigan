import { describe, expect, it } from "vitest";
import { buildingSpecSchema, siftProjectV2Schema, type DesignEdge } from "./contracts";
import { validateConnection, validateGraph } from "./graph";
import { buildingSpecFromMassing, migrateProject, reconcileStores, toLegacyProject } from "./migrate";
import { sampleProjects } from "./samples";

const validSpec = () => buildingSpecFromMassing("Test", sampleProjects[0].massing);
const n = (id: string, type: "prompt" | "generation" | "variation" | "model" | "render") => ({ id, type });
const e = (source: string, sourcePort: string, target: string, targetPort: string): DesignEdge => ({ id: `${source}-${target}`, source, sourcePort, target, targetPort });

describe("BuildingSpec", () => {
  it("accepts a migrated spec", () => expect(buildingSpecSchema.safeParse(validSpec()).success).toBe(true));
  it("rejects unknown materials, duplicate ids, and lone setback fields", () => {
    const spec = validSpec();
    const bad = { ...spec, volumes: [{ ...spec.volumes[0], materialId: "gold", setbackEvery: 3, setbackAmount: undefined }, { ...spec.volumes[0] }] };
    const messages = (buildingSpecSchema.safeParse(bad).error?.issues ?? []).map((issue) => issue.message).join(" | ");
    expect(messages).toMatch(/Unknown material/);
    expect(messages).toMatch(/Duplicate volume id/);
    expect(messages).toMatch(/provided together/);
  });
  it("rejects non-positive dimensions and over-tall volumes", () => {
    const spec = validSpec();
    expect(buildingSpecSchema.safeParse({ ...spec, footprint: { type: "circle", radius: 0 } }).success).toBe(false);
    expect(buildingSpecSchema.safeParse({ ...spec, volumes: [{ ...spec.volumes[0], startFloor: 100, floorCount: 30 }] }).success).toBe(false);
  });
});

describe("graph validation", () => {
  const nodes = [n("p", "prompt"), n("g", "generation"), n("v", "variation"), n("m", "model"), n("r", "render")];
  it("accepts the artifact flow prompt -> generation -> variation -> model/render", () => {
    const edges = [e("p", "prompt", "g", "prompt"), e("g", "spec", "v", "spec"), e("v", "spec", "m", "spec"), e("v", "spec", "r", "model")];
    expect(validateGraph(nodes, edges).ok).toBe(true);
  });
  it("rejects type mismatches, unknown ports, self loops, and missing nodes", () => {
    expect(validateConnection(nodes, [], { source: "p", sourcePort: "prompt", target: "m", targetPort: "spec" })).toMatchObject({ ok: false, code: "type-mismatch" });
    expect(validateConnection(nodes, [], { source: "g", sourcePort: "nope", target: "m", targetPort: "spec" })).toMatchObject({ ok: false, code: "unknown-port" });
    expect(validateConnection(nodes, [], { source: "g", sourcePort: "spec", target: "g", targetPort: "prompt" })).toMatchObject({ ok: false, code: "self-loop" });
    expect(validateConnection(nodes, [], { source: "x", sourcePort: "spec", target: "m", targetPort: "spec" })).toMatchObject({ ok: false, code: "missing-node" });
  });
  it("rejects duplicate inputs and cycles", () => {
    const base = [e("g", "spec", "v1", "spec"), e("v1", "spec", "v2", "spec")];
    const vn = [n("g", "generation"), n("v1", "variation"), n("v2", "variation")];
    expect(validateConnection(vn, base, { source: "v2", sourcePort: "spec", target: "v1", targetPort: "spec" })).toMatchObject({ ok: false, code: "input-occupied" });
    const loop = [n("a", "variation"), n("b", "variation")];
    expect(validateConnection(loop, [e("a", "spec", "b", "spec")], { source: "b", sourcePort: "spec", target: "a", targetPort: "spec" })).toMatchObject({ ok: false, code: "cycle" });
  });
});

describe("migration v1 -> v2", () => {
  it.each(sampleProjects)("migrates $name and round-trips losslessly", (sample) => {
    const result = migrateProject(sample);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.warnings).toEqual([]);
    expect(siftProjectV2Schema.safeParse(result.project).success).toBe(true);
    expect(toLegacyProject(result.project)).toEqual(sample);
  });

  it("keeps jobs and artifacts separate and retains provider task metadata", () => {
    const sample = { ...sampleProjects[0], provider: "meshy" as const, providerTask: { id: "t-1", status: "IN_PROGRESS" as const, progress: 40 } };
    const result = migrateProject(sample);
    if (!result.ok) throw new Error(result.error);
    const [job] = Object.values(result.project.jobs);
    expect(job).toMatchObject({ providerTaskId: "t-1", status: "running", progress: 40 });
    expect(job.resultArtifactId).toBeUndefined();
    expect(Object.values(result.project.artifacts)[0].kind).toBe("building-spec");
    expect(toLegacyProject(result.project)).toEqual(sample);
  });

  it("keeps the project but reports invalid legacy wiring", () => {
    const sample = structuredClone(sampleProjects[0]);
    sample.graph.edges.push({ id: "bad", source: "export", target: "brief" });
    const result = migrateProject(sample);
    if (!result.ok) throw new Error(result.error);
    expect(result.warnings.join()).toMatch(/Dropped edge "bad"/);
    expect(result.project.graph.edges).toHaveLength(3);
  });

  it("fails clearly on unrecognized records", () => {
    expect(migrateProject({ schemaVersion: 7 })).toMatchObject({ ok: false });
    expect(migrateProject(null)).toMatchObject({ ok: false });
  });
});

describe("store reconciliation", () => {
  it("merges legacy records, prefers v2 on id clash, and preserves unreadable v2 records", () => {
    const [a, b] = sampleProjects;
    const v2a = migrateProject({ ...a, name: "Renamed" });
    if (!v2a.ok) throw new Error(v2a.error);
    const junk = { schemaVersion: 99, keep: "me" };
    const { projects, preserved } = reconcileStores([v2a.project, junk], [a, b, { broken: true }]);
    expect(projects.map((p) => p.id).sort()).toEqual([a.id, b.id].sort());
    expect(projects.find((p) => p.id === a.id)?.name).toBe("Renamed");
    expect(preserved).toEqual([junk]);
  });
  it("handles empty stores", () => expect(reconcileStores(undefined, undefined)).toEqual({ projects: [], preserved: [] }));
});

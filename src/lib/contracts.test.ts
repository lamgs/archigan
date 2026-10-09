import { describe, expect, it } from "vitest";
import { generateRequestSchema, isSupportedProvider, SUPPORTED_PROVIDERS, siftProjectSchema, siftProjectV2Schema } from "./contracts";
import { migrateProject, reconcileStores } from "./migrate";
import { createWorkflowProject } from "./projects";
import { legacySamples as sampleProjects } from "./legacy-fixtures";

describe("contracts", () => {
  it("accepts every bundled sample", () => {
    sampleProjects.forEach((project) => expect(siftProjectSchema.parse(project)).toEqual(project));
  });

  it("rejects unknown providers", () => {
    expect(generateRequestSchema.safeParse({ prompt: "A library", refinement: "", provider: "browser-key" }).success).toBe(false);
  });
});

describe("provider enum backward compatibility (ADR-017)", () => {
  it("still accepts the original values and every newer hosted provider for settings and jobs", async () => {
    const { projectSettingsSchema, providerSchema } = await import("./contracts");
    for (const provider of ["procedural", "meshy", "tripo", "hunyuan3d-rapid", "hunyuan3d-pro"]) {
      expect(providerSchema.safeParse(provider).success).toBe(true);
      expect(projectSettingsSchema.safeParse({ provider }).success).toBe(true);
    }
    expect(providerSchema.safeParse("rodin").success).toBe(false);
  });
  it("still parses generate requests carrying `meshy` or `procedural`", () => {
    for (const provider of ["procedural", "meshy"]) expect(generateRequestSchema.safeParse({ prompt: "A library", provider, confirmSpend: provider === "meshy" }).success).toBe(true);
  });
});

describe("removed providers (ADR-018): legacy projects still open, as procedural", () => {
  const base = createWorkflowProject({ id: "legacy", name: "Legacy", now: "2026-10-09T10:00:00.000Z", prompt: "A tower", refinement: "" });
  const saved = (provider: string, jobProvider?: string) => JSON.parse(JSON.stringify({
    ...base,
    settings: { ...base.settings, provider },
    jobs: jobProvider ? { j1: { id: "j1", nodeId: "legacy-generation", provider: jobProvider, providerTaskId: "task-123456", status: "running", progress: 10 } } : {},
  }));

  it("defines the supported set", () => {
    expect([...SUPPORTED_PROVIDERS]).toEqual(["procedural", "tripo"]);
    expect(["procedural", "tripo"].every(isSupportedProvider)).toBe(true);
    expect(["meshy", "hunyuan3d-rapid", "hunyuan3d-pro", "tencent-rapid", "tencent-pro", "rodin", undefined].some(isSupportedProvider)).toBe(false);
  });
  it("still parses a project saved with provider meshy, then loads it as procedural", () => {
    const raw = saved("meshy");
    expect(siftProjectV2Schema.safeParse(raw).success).toBe(true);
    const result = migrateProject(raw);
    expect(result).toMatchObject({ ok: true, project: { settings: { provider: "procedural" } } });
    expect(result.ok && result.warnings.join(" ")).toMatch(/no longer supported/);
  });
  it("coerces every removed id and keeps supported ones", () => {
    for (const id of ["meshy", "hunyuan3d-rapid", "hunyuan3d-pro", "tencent-rapid", "tencent-pro"]) expect(migrateProject(saved(id))).toMatchObject({ ok: true, project: { settings: { provider: "procedural" } } });
    expect(migrateProject(saved("tripo"))).toMatchObject({ ok: true, project: { settings: { provider: "tripo" } }, warnings: [] });
  });
  it("keeps a removed provider's job as history while the project provider is coerced", () => {
    const result = migrateProject(saved("hunyuan3d-pro", "hunyuan3d-pro"));
    if (!result.ok) throw new Error(result.error);
    expect(result).toMatchObject({ ok: true, project: { settings: { provider: "procedural" }, jobs: { j1: { provider: "hunyuan3d-pro", status: "running", providerTaskId: "task-123456" } } } });
  });
  it("coerces through the storage read path (reconcileStores) and for legacy v1 records", () => {
    const store = reconcileStores([saved("meshy")], []);
    expect(store.preserved).toEqual([]);
    expect(store.projects[0].settings.provider).toBe("procedural");
    const v1 = { ...sampleProjects[0], provider: "meshy" as const };
    expect(reconcileStores([], [v1]).projects[0].settings.provider).toBe("procedural");
  });
});

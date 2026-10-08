import { IDBFactory } from "fake-indexeddb";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { siftProjectSchema } from "./contracts";
import { copyFromSample, createBlankProject, isBlankProject, renameProject, uniqueName, validateProjectName } from "./projects";
import { sampleProjects } from "./samples";

const NOW = "2026-10-09T10:00:00.000Z";

describe("project helpers", () => {
  it("validates and normalizes names", () => {
    expect(validateProjectName("  My   study ")).toEqual({ ok: true, name: "My study" });
    expect(validateProjectName("   ")).toMatchObject({ ok: false });
    expect(validateProjectName("x".repeat(81))).toMatchObject({ ok: false });
  });
  it("generates unique names case-insensitively", () => {
    expect(uniqueName("Untitled study", [])).toBe("Untitled study");
    expect(uniqueName("Untitled study", ["untitled study", "Untitled study 2"])).toBe("Untitled study 3");
  });
  it("creates blank projects that are blank until a prompt exists", () => {
    const blank = createBlankProject("a", NOW, ["Untitled study"]);
    expect(blank.name).toBe("Untitled study 2");
    expect(isBlankProject(blank)).toBe(true);
    expect(siftProjectSchema.safeParse(blank).success).toBe(false); // not persistable yet
    expect(siftProjectSchema.safeParse({ ...blank, prompt: "A tower" }).success).toBe(true);
  });
  it("copies samples without touching the original", () => {
    const copy = copyFromSample(sampleProjects[0], "new-id", NOW, [sampleProjects[0].name]);
    expect(copy.id).toBe("new-id");
    expect(copy.name).toBe(`${sampleProjects[0].name} 2`);
    expect(sampleProjects[0].id).toBe("sample-courtyard");
    expect(siftProjectSchema.safeParse(copy).success).toBe(true);
  });
  it("renames immutably and rejects invalid names", () => {
    const original = sampleProjects[1];
    const result = renameProject(original, "  Tower B ", NOW);
    expect(result).toMatchObject({ ok: true, project: { name: "Tower B", updatedAt: NOW } });
    expect(original.name).toBe("Spiral Habitat");
    expect(renameProject(original, "", NOW)).toMatchObject({ ok: false });
  });
});

describe("storage CRUD (fake IndexedDB)", () => {
  beforeEach(() => {
    globalThis.indexedDB = new IDBFactory();
    vi.resetModules();
  });
  const mk = (id: string, name: string, updatedAt: string) => ({ ...sampleProjects[0], id, name, updatedAt });

  it("saves, renames, lists newest first, and survives a reload", async () => {
    const storage = await import("./storage");
    await storage.saveProject(mk("a", "Alpha", "2026-10-01T00:00:00.000Z"));
    await storage.saveProject(mk("b", "Beta", "2026-10-02T00:00:00.000Z"));
    await storage.saveProject(mk("a", "Alpha renamed", "2026-10-03T00:00:00.000Z"));
    vi.resetModules();
    const reloaded = await import("./storage");
    expect((await reloaded.listProjects()).map((p) => p.name)).toEqual(["Alpha renamed", "Beta"]);
  });

  it("deletes, and deletion persists across reloads", async () => {
    const storage = await import("./storage");
    await storage.saveProject(mk("a", "Alpha", "2026-10-01T00:00:00.000Z"));
    await storage.saveProject(mk("b", "Beta", "2026-10-02T00:00:00.000Z"));
    expect((await storage.deleteProject("a")).map((p) => p.id)).toEqual(["b"]);
    vi.resetModules();
    expect((await (await import("./storage")).listProjects()).map((p) => p.id)).toEqual(["b"]);
  });

  it("keeps deleted legacy (v1-key) projects deleted and revives them if re-saved", async () => {
    const { createStore, set } = await import("idb-keyval");
    await set("projects-v1", [mk("legacy", "Legacy", "2026-09-01T00:00:00.000Z")], createStore("sift-projects", "projects"));
    const storage = await import("./storage");
    expect((await storage.listProjects()).map((p) => p.id)).toEqual(["legacy"]);
    expect(await storage.deleteProject("legacy")).toEqual([]);
    expect((await storage.saveProject(mk("legacy", "Back", "2026-10-05T00:00:00.000Z"))).map((p) => p.name)).toEqual(["Back"]);
  });

  it("does not lose projects under concurrent saves", async () => {
    const storage = await import("./storage");
    await Promise.all(["a", "b", "c", "d"].map((id, i) => storage.saveProject(mk(id, id, `2026-10-0${i + 1}T00:00:00.000Z`))));
    expect((await storage.listProjects()).map((p) => p.id).sort()).toEqual(["a", "b", "c", "d"]);
  });
});

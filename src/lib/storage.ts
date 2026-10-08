import { createStore, del, get, set, update } from "idb-keyval";
import { siftProjectV2Schema, type SiftProjectV2 } from "./contracts";
import { reconcileStores } from "./migrate";

const store = createStore("sift-projects", "projects");
const assetStore = createStore("sift-assets", "assets"); // binary artifacts (render PNGs); keyed by Artifact.storageKey
const LEGACY_KEY = "projects-v1"; // read-only; never rewritten or deleted
const PROJECTS_KEY = "projects-v2";
const DELETED_KEY = "projects-deleted"; // tombstones so deleted legacy projects stay deleted
const MAX_PROJECTS = 30;

const byRecency = <T extends { updatedAt: string }>(a: T, b: T) => b.updatedAt.localeCompare(a.updatedAt);

async function readAll() {
  const [v2Raw, legacyRaw, deleted] = await Promise.all([get(PROJECTS_KEY, store), get(LEGACY_KEY, store), get<string[]>(DELETED_KEY, store)]);
  return { v2Raw, legacyRaw, deleted: Array.isArray(deleted) ? deleted : [] };
}

export async function listProjects(): Promise<SiftProjectV2[]> {
  const { v2Raw, legacyRaw, deleted } = await readAll();
  return reconcileStores(v2Raw, legacyRaw, deleted).projects.sort(byRecency);
}

export async function saveProject(project: SiftProjectV2) {
  const parsed = siftProjectV2Schema.parse(project);
  // Saving a previously deleted id (e.g. re-saving an open project) revives it.
  await update<string[]>(DELETED_KEY, (current) => (Array.isArray(current) ? current.filter((item) => item !== parsed.id) : []), store);
  const { legacyRaw, deleted } = await readAll();
  // Single read-modify-write transaction: concurrent saves cannot overwrite one another,
  // and unreadable records already in the v2 store are written back rather than dropped.
  await update<unknown[]>(
    PROJECTS_KEY,
    (current) => {
      const { projects, preserved } = reconcileStores(current, legacyRaw, deleted);
      const next = [parsed, ...projects.filter((item) => item.id !== parsed.id).sort(byRecency)].slice(0, MAX_PROJECTS);
      return [...next, ...preserved];
    },
    store,
  );
  return listProjects();
}

export async function deleteProject(id: string) {
  const doomed = (await listProjects()).find((project) => project.id === id);
  const assetKeys = doomed ? Object.values(doomed.artifacts).filter((artifact) => artifact.kind === "render-png").map((artifact) => artifact.storageKey) : [];
  // Tombstone first: if the second write fails the project is still hidden rather than half-deleted.
  await update<string[]>(DELETED_KEY, (current) => [...new Set([...(Array.isArray(current) ? current : []), id])], store);
  await update<unknown[]>(PROJECTS_KEY, (current) => (Array.isArray(current) ? current.filter((item) => (item as { id?: unknown } | null)?.id !== id) : []), store);
  await deleteAssets(assetKeys);
  return listProjects();
}

type StoredAsset = { type: string; bytes: ArrayBuffer };

/** Persists a binary asset. Stored as bytes + type so it round-trips in every IndexedDB implementation. */
export async function saveAsset(key: string, blob: Blob) {
  const asset: StoredAsset = { type: blob.type || "application/octet-stream", bytes: await blob.arrayBuffer() };
  await set(key, asset, assetStore);
}

export async function loadAsset(key: string): Promise<Blob | null> {
  const asset = await get<StoredAsset>(key, assetStore);
  return asset && asset.bytes instanceof ArrayBuffer ? new Blob([asset.bytes], { type: asset.type }) : null;
}

export async function deleteAssets(keys: string[]) {
  await Promise.all(keys.map((key) => del(key, assetStore)));
}

/** Checks that IndexedDB actually works (it can be blocked, full, or absent in private/locked-down browsers). */
export async function probeStorage(timeoutMs = 3000): Promise<boolean> {
  try {
    const attempt = (async () => {
      await set("__probe__", Date.now(), store);
      const value = await get("__probe__", store);
      await del("__probe__", store);
      return typeof value === "number";
    })();
    return await Promise.race([attempt, new Promise<boolean>((resolve) => setTimeout(() => resolve(false), timeoutMs))]);
  } catch {
    return false;
  }
}

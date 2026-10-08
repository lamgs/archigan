import { createStore, get, update } from "idb-keyval";
import { siftProjectV2Schema, type SiftProjectV2 } from "./contracts";
import { reconcileStores } from "./migrate";

const store = createStore("sift-projects", "projects");
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
  // Tombstone first: if the second write fails the project is still hidden rather than half-deleted.
  await update<string[]>(DELETED_KEY, (current) => [...new Set([...(Array.isArray(current) ? current : []), id])], store);
  await update<unknown[]>(PROJECTS_KEY, (current) => (Array.isArray(current) ? current.filter((item) => (item as { id?: unknown } | null)?.id !== id) : []), store);
  return listProjects();
}

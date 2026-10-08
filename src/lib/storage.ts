import { createStore, get, update } from "idb-keyval";
import { siftProjectSchema, type SiftProject } from "./contracts";
import { migrateProject, reconcileStores, toLegacyProject } from "./migrate";

const store = createStore("sift-projects", "projects");
const LEGACY_KEY = "projects-v1"; // read-only; never rewritten or deleted
const PROJECTS_KEY = "projects-v2";
const MAX_PROJECTS = 30;

export async function listProjects(): Promise<SiftProject[]> {
  const [v2Raw, legacyRaw] = await Promise.all([get(PROJECTS_KEY, store), get(LEGACY_KEY, store)]);
  return reconcileStores(v2Raw, legacyRaw)
    .projects.flatMap((project) => {
      const legacy = toLegacyProject(project);
      return legacy ? [legacy] : [];
    })
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export async function saveProject(project: SiftProject) {
  const parsed = siftProjectSchema.parse(project);
  const migrated = migrateProject(parsed);
  if (!migrated.ok) throw new Error(migrated.error);
  const legacyRaw = await get(LEGACY_KEY, store);
  // Single read-modify-write transaction: concurrent saves cannot overwrite one another,
  // and unreadable records already in the v2 store are written back rather than dropped.
  await update<unknown[]>(
    PROJECTS_KEY,
    (current) => {
      const { projects, preserved } = reconcileStores(current, legacyRaw);
      const next = [migrated.project, ...projects.filter((item) => item.id !== migrated.project.id).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))].slice(0, MAX_PROJECTS);
      return [...next, ...preserved];
    },
    store,
  );
  return listProjects();
}

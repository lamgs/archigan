import { createStore, get, set } from "idb-keyval";
import { siftProjectSchema, type SiftProject } from "./contracts";

const store = createStore("sift-projects", "projects");
const PROJECTS_KEY = "projects-v1";

export async function listProjects(): Promise<SiftProject[]> {
  const raw = await get(PROJECTS_KEY, store);
  if (!Array.isArray(raw)) return [];
  return raw.flatMap((item) => {
    const result = siftProjectSchema.safeParse(item);
    return result.success ? [result.data] : [];
  });
}

export async function saveProject(project: SiftProject) {
  const parsed = siftProjectSchema.parse(project);
  const current = await listProjects();
  const next = [parsed, ...current.filter((item) => item.id !== parsed.id)].slice(0, 30);
  await set(PROJECTS_KEY, next, store);
  return next;
}


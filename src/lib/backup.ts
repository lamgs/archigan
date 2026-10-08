import { siftProjectV2Schema, type SiftProjectV2 } from "./contracts";
import { migrateProject } from "./migrate";

export const BACKUP_FORMAT = "sift-project-backup";

/**
 * Portable backup of one project (graph, specs, revisions, jobs, settings). Binary assets (render PNGs, hosted GLBs)
 * live in IndexedDB and are NOT included; their artifacts are listed so the UI can say which were left out.
 */
export function exportProjectJson(project: SiftProjectV2): string {
  return JSON.stringify({ format: BACKUP_FORMAT, version: 1, exportedAt: new Date().toISOString(), project }, null, 2);
}

export function backupFilename(project: Pick<SiftProjectV2, "name">) {
  return `${project.name.replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "sift-project"}.sift.json`;
}

export function missingAssets(project: SiftProjectV2) {
  return Object.values(project.artifacts).filter((artifact) => artifact.kind !== "building-spec").length;
}

export type ImportResult = { ok: true; project: SiftProjectV2; warnings: string[] } | { ok: false; error: string };

/** Parses a backup file (or a bare project record, v1 or v2) without trusting it. */
export function parseProjectJson(text: string, newId: () => string, takenIds: string[] = []): ImportResult {
  if (text.length > 20_000_000) return { ok: false, error: "That file is too large to be a Sift project." };
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return { ok: false, error: "That file is not valid JSON." };
  }
  const record = raw && typeof raw === "object" && (raw as { format?: unknown }).format === BACKUP_FORMAT ? (raw as { project?: unknown }).project : raw;
  const migrated = migrateProject(record);
  if (!migrated.ok) return { ok: false, error: `This is not a Sift project backup. ${migrated.error}` };
  const warnings = [...migrated.warnings];
  let project = migrated.project;
  if (takenIds.includes(project.id)) {
    project = { ...project, id: newId() };
    warnings.push("A project with this id already exists, so the import was saved as a copy.");
  }
  const lost = missingAssets(project);
  if (lost > 0) warnings.push(`${lost} image/model file${lost === 1 ? "" : "s"} are not included in backups; re-run renders or hosted generations if you need them.`);
  const check = siftProjectV2Schema.safeParse(project);
  return check.success ? { ok: true, project: check.data, warnings } : { ok: false, error: check.error.issues[0]?.message ?? "The project data is invalid." };
}

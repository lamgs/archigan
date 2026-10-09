import {
  buildingSpecSchema,
  massingSpecSchema,
  siftProjectSchema,
  siftProjectV2Schema,
  isSupportedProvider,
  type BuildingSpec,
  type DesignEdge,
  type DesignNode,
  type DesignNodeType,
  type GenerationJob,
  type MassingSpec,
  type SiftProject,
  type SiftProjectV2,
} from "./contracts";
import { NODE_PORTS, validateGraph } from "./graph";

const V1_TO_V2_NODE: Record<SiftProject["graph"]["nodes"][number]["type"], DesignNodeType> = {
  brief: "prompt",
  massing: "generation",
  refine: "variation",
  export: "model",
};
const V2_TO_V1_NODE: Partial<Record<DesignNodeType, SiftProject["graph"]["nodes"][number]["type"]>> = {
  prompt: "brief",
  generation: "massing",
  variation: "refine",
  model: "export",
};
const JOB_STATUS = { PENDING: "queued", IN_PROGRESS: "running", SUCCEEDED: "completed", FAILED: "failed", CANCELED: "cancelled" } as const;
const JOB_STATUS_BACK = { queued: "PENDING", running: "IN_PROGRESS", completed: "SUCCEEDED", failed: "FAILED", cancelled: "CANCELED" } as const;

const MATERIALS: Record<MassingSpec["material"], BuildingSpec["materials"][string]> = {
  limestone: { kind: "concrete", color: "#d8cfbb" },
  terracotta: { kind: "clay", color: "#b5543a" },
  concrete: { kind: "concrete", color: "#9a9a96" },
  glass: { kind: "glass", color: "#9cc4d4" },
};

/** Approximate canonical spec for a legacy massing study. The exact legacy values travel in artifact metadata. */
export function buildingSpecFromMassing(name: string, massing: MassingSpec): BuildingSpec {
  const stepped = massing.terrace > 0.02;
  return buildingSpecSchema.parse({
    schemaVersion: 1,
    name: name.slice(0, 80) || "Untitled study",
    units: "m",
    floorHeight: massing.floorHeight,
    footprint: { type: "rectangle", width: massing.width, depth: massing.depth },
    volumes: [
      {
        id: "tower",
        role: "tower",
        startFloor: 0,
        floorCount: massing.floors,
        footprintScale: 1,
        offsetX: 0,
        offsetZ: 0,
        rotationDegrees: Math.max(-180, Math.min(180, ((massing.twist * massing.floors) * 180) / Math.PI)),
        taper: 0,
        ...(stepped ? { setbackEvery: 4, setbackAmount: Math.min(20, Math.round(massing.terrace * massing.width * 10) / 10) } : {}),
        materialId: massing.material,
      },
    ],
    facade: { style: massing.material === "glass" ? "grid" : "horizontal", glazingRatio: massing.material === "glass" ? 0.8 : 0.35 },
    roof: { style: stepped ? "terrace" : "flat" },
    materials: { [massing.material]: MATERIALS[massing.material] },
  });
}

export type MigrationResult = { ok: true; project: SiftProjectV2; warnings: string[] } | { ok: false; error: string };

export function migrateV1ToV2(v1: SiftProject): { project: SiftProjectV2; warnings: string[] } {
  const warnings: string[] = [];
  const nodes: DesignNode[] = v1.graph.nodes.map((node) => ({ id: node.id, type: V1_TO_V2_NODE[node.type], position: node.position, params: {} }));
  const find = (type: DesignNodeType) => nodes.find((node) => node.type === type);

  const generation = find("generation");
  const artifacts: SiftProjectV2["artifacts"] = {};
  const jobs: SiftProjectV2["jobs"] = {};
  const sourceNodeId = generation?.id ?? nodes[0]?.id ?? "legacy";
  const artifactId = `${v1.id}-spec-1`;
  artifacts[artifactId] = {
    id: artifactId,
    kind: "building-spec",
    sourceNodeId,
    createdAt: v1.updatedAt,
    storageKey: `inline:${artifactId}`,
    metadata: { spec: buildingSpecFromMassing(v1.name, v1.massing), legacyMassing: v1.massing, origin: "migrated-v1" },
  };
  if (generation) generation.artifactId = artifactId;

  const promptNode = find("prompt");
  if (promptNode) promptNode.params = { text: v1.prompt };
  const variation = find("variation");
  if (variation) variation.params = { text: v1.refinement };

  if (generation) {
    const jobId = `${v1.id}-job-1`;
    const task = v1.providerTask;
    const job: GenerationJob = task
      ? {
          id: jobId,
          nodeId: generation.id,
          provider: v1.provider,
          providerTaskId: task.id,
          status: JOB_STATUS[task.status],
          ...(task.progress !== undefined ? { progress: task.progress } : {}),
          ...(task.glbUrl ? { outputUrl: task.glbUrl } : {}),
        }
      : { id: jobId, nodeId: generation.id, provider: v1.provider, status: "completed", resultArtifactId: artifactId };
    jobs[jobId] = job;
  }

  const edges: DesignEdge[] = [];
  v1.graph.edges.forEach((edge) => {
    const source = nodes.find((node) => node.id === edge.source);
    const target = nodes.find((node) => node.id === edge.target);
    if (!source || !target) return void warnings.push(`Dropped edge "${edge.id}": missing node.`);
    edges.push({ id: edge.id, source: source.id, sourcePort: NODE_PORTS[source.type].outputs[0]?.id ?? "", target: target.id, targetPort: NODE_PORTS[target.type].inputs[0]?.id ?? "" });
  });
  const checked = validateGraph(nodes, edges);
  checked.errors.forEach((error) => warnings.push(`Dropped edge "${error.edgeId}": ${error.message}`));

  const project = siftProjectV2Schema.parse({
    schemaVersion: 2,
    id: v1.id,
    name: v1.name,
    createdAt: v1.createdAt,
    updatedAt: v1.updatedAt,
    viewport: { x: 0, y: 0, zoom: 1 },
    graph: { nodes, edges: checked.accepted },
    artifacts,
    jobs,
    revisions: {},
    settings: { provider: v1.provider },
  });
  return { project, warnings };
}

/**
 * ADR-018: projects saved with a provider the product no longer offers open as `procedural`. Only `settings.provider`
 * is rewritten; jobs keep their raw provider value as history.
 */
export function coerceProjectProvider(project: SiftProjectV2): { project: SiftProjectV2; warnings: string[] } {
  if (isSupportedProvider(project.settings.provider)) return { project, warnings: [] };
  return { project: { ...project, settings: { ...project.settings, provider: "procedural" } }, warnings: [`The provider "${project.settings.provider}" is no longer supported; this project now uses Local procedural.`] };
}

/** Parses any persisted record (v1 or v2) into v2 without throwing. */
export function migrateProject(raw: unknown): MigrationResult {
  const v2 = siftProjectV2Schema.safeParse(raw);
  if (v2.success) return { ok: true, ...coerceProjectProvider(v2.data) };
  const v1 = siftProjectSchema.safeParse(raw);
  if (v1.success) {
    const migrated = migrateV1ToV2(v1.data);
    const coerced = coerceProjectProvider(migrated.project);
    return { ok: true, project: coerced.project, warnings: [...migrated.warnings, ...coerced.warnings] };
  }
  const version = (raw as { schemaVersion?: unknown } | null)?.schemaVersion;
  return { ok: false, error: `Unrecognized project record (schemaVersion ${String(version)}): ${(version === 2 ? v2 : v1).error.issues[0]?.message ?? "invalid"}` };
}

/** Projects the v2 aggregate back to the v1 shape the current UI renders. Returns null if it cannot be represented. */
export function toLegacyProject(project: SiftProjectV2): SiftProject | null {
  const generation = project.graph.nodes.find((node) => node.type === "generation");
  const artifact = generation?.artifactId ? project.artifacts[generation.artifactId] : undefined;
  const massing = massingSpecSchema.safeParse(artifact?.metadata.legacyMassing);
  if (!massing.success) return null;
  const nodes = project.graph.nodes.flatMap((node) => {
    const type = V2_TO_V1_NODE[node.type];
    return type ? [{ id: node.id, type, position: node.position }] : [];
  });
  const ids = new Set(nodes.map((node) => node.id));
  const job = generation ? Object.values(project.jobs).find((item) => item.nodeId === generation.id && item.providerTaskId) : undefined;
  const text = (type: DesignNodeType) => {
    const value = project.graph.nodes.find((node) => node.type === type)?.params.text;
    return typeof value === "string" ? value : "";
  };
  const legacy = {
    schemaVersion: 1 as const,
    id: project.id,
    name: project.name,
    createdAt: project.createdAt,
    updatedAt: project.updatedAt,
    prompt: text("prompt"),
    refinement: text("variation"),
    provider: project.settings.provider,
    massing: massing.data,
    graph: { nodes, edges: project.graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)).map((edge) => ({ id: edge.id, source: edge.source, target: edge.target })) },
    ...(job?.providerTaskId && job.status in JOB_STATUS_BACK
      ? { providerTask: { id: job.providerTaskId, status: JOB_STATUS_BACK[job.status as keyof typeof JOB_STATUS_BACK], ...(job.progress !== undefined ? { progress: job.progress } : {}), ...(job.outputUrl ? { glbUrl: job.outputUrl } : {}) } }
      : {}),
  };
  const parsed = siftProjectSchema.safeParse(legacy);
  return parsed.success ? parsed.data : null;
}

export type ReconciledStore = { projects: SiftProjectV2[]; preserved: unknown[] };

/**
 * Merges the current v2 store with the untouched legacy v1 store. Valid records are migrated; records that
 * cannot be parsed are returned in `preserved` so callers write them back instead of discarding them.
 * v2 records win over legacy records with the same id. `deletedIds` are tombstones that keep deleted legacy
 * projects from reappearing (the legacy key is never rewritten).
 */
export function reconcileStores(v2Raw: unknown, legacyRaw: unknown, deletedIds: string[] = []): ReconciledStore {
  const deleted = new Set(deletedIds);
  const projects = new Map<string, SiftProjectV2>();
  const preserved: unknown[] = [];
  const ingest = (raw: unknown, overwrite: boolean) => {
    if (!Array.isArray(raw)) return;
    raw.forEach((item) => {
      const result = migrateProject(item);
      if (!result.ok) return void preserved.push(item);
      if (deleted.has(result.project.id)) return;
      if (overwrite || !projects.has(result.project.id)) projects.set(result.project.id, result.project);
    });
  };
  ingest(legacyRaw, false);
  ingest(v2Raw, true);
  // Legacy-key rejects stay in the legacy key (never deleted); only v2-key rejects need carrying forward.
  const v2Preserved = Array.isArray(v2Raw) ? preserved.filter((item) => (v2Raw as unknown[]).includes(item)) : [];
  return { projects: [...projects.values()], preserved: v2Preserved };
}

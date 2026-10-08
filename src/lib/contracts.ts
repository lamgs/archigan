import { z } from "zod";

export const providerSchema = z.enum(["procedural", "meshy"]);
export type Provider = z.infer<typeof providerSchema>;

export const massingSpecSchema = z.object({
  seed: z.number().int().nonnegative(),
  floors: z.number().int().min(2).max(42),
  width: z.number().min(8).max(80),
  depth: z.number().min(8).max(80),
  floorHeight: z.number().min(2.5).max(6),
  twist: z.number().min(-0.35).max(0.35),
  terrace: z.number().min(0).max(0.12),
  courtyard: z.boolean(),
  material: z.enum(["limestone", "terracotta", "concrete", "glass"]),
});
export type MassingSpec = z.infer<typeof massingSpecSchema>;

export const graphNodeSchema = z.object({
  id: z.string(),
  type: z.enum(["brief", "massing", "refine", "export"]),
  position: z.object({ x: z.number(), y: z.number() }),
});

export const graphEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
});

export const providerTaskSchema = z.object({
  id: z.string(),
  status: z.enum(["PENDING", "IN_PROGRESS", "SUCCEEDED", "FAILED", "CANCELED"]),
  progress: z.number().min(0).max(100).optional(),
  glbUrl: z.string().url().optional(),
});

export const siftProjectSchema = z.object({
  schemaVersion: z.literal(1),
  id: z.string(),
  name: z.string().min(1).max(80),
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  prompt: z.string().min(1).max(800),
  refinement: z.string().max(400),
  provider: providerSchema,
  massing: massingSpecSchema,
  graph: z.object({
    nodes: z.array(graphNodeSchema),
    edges: z.array(graphEdgeSchema),
  }),
  providerTask: providerTaskSchema.optional(),
});
export type SiftProject = z.infer<typeof siftProjectSchema>;

// ---------------------------------------------------------------------------
// Schema v2 — canonical contracts (P0.09). v1 above remains the UI-facing shape
// until later tasks move the shell onto v2; storage migrates v1 -> v2.
// ---------------------------------------------------------------------------

const id = z.string().min(1).max(120);
const isoDate = z.string().datetime();
const finite = z.number().finite();

export const materialSchema = z.object({
  kind: z.enum(["clay", "concrete", "glass", "metal"]),
  color: z.string().regex(/^#[0-9a-fA-F]{6}$/),
});

export const volumeSchema = z.object({
  id,
  role: z.enum(["podium", "tower", "wing", "core"]),
  startFloor: z.number().int().min(0).max(120),
  floorCount: z.number().int().min(1).max(120),
  footprintScale: z.number().min(0.1).max(2),
  offsetX: z.number().min(-80).max(80),
  offsetZ: z.number().min(-80).max(80),
  rotationDegrees: z.number().min(-180).max(180),
  taper: z.number().min(-0.5).max(0.9),
  setbackEvery: z.number().int().min(1).max(60).optional(),
  setbackAmount: z.number().min(0).max(20).optional(),
  materialId: z.string().min(1),
});

export const MAX_BUILDING_FLOORS = 120;
export const MAX_BUILDING_VOLUMES = 12;

export const buildingSpecSchema = z
  .object({
    schemaVersion: z.literal(1),
    name: z.string().min(1).max(80),
    units: z.literal("m"),
    floorHeight: z.number().min(2.5).max(8),
    footprint: z.discriminatedUnion("type", [
      z.object({ type: z.literal("rectangle"), width: z.number().min(4).max(200), depth: z.number().min(4).max(200) }),
      z.object({ type: z.literal("circle"), radius: z.number().min(2).max(100) }),
    ]),
    volumes: z.array(volumeSchema).min(1).max(MAX_BUILDING_VOLUMES),
    facade: z.object({ style: z.enum(["solid", "horizontal", "vertical", "grid"]), glazingRatio: z.number().min(0).max(1) }),
    roof: z.object({ style: z.enum(["flat", "terrace", "crown"]) }),
    materials: z.record(z.string(), materialSchema),
  })
  .superRefine((spec, ctx) => {
    const seen = new Set<string>();
    spec.volumes.forEach((volume, index) => {
      if (seen.has(volume.id)) ctx.addIssue({ code: "custom", path: ["volumes", index, "id"], message: `Duplicate volume id "${volume.id}".` });
      seen.add(volume.id);
      if (!(volume.materialId in spec.materials)) ctx.addIssue({ code: "custom", path: ["volumes", index, "materialId"], message: `Unknown material "${volume.materialId}".` });
      if ((volume.setbackEvery === undefined) !== (volume.setbackAmount === undefined)) ctx.addIssue({ code: "custom", path: ["volumes", index, "setbackEvery"], message: "setbackEvery and setbackAmount must be provided together." });
      if (volume.startFloor + volume.floorCount > MAX_BUILDING_FLOORS) ctx.addIssue({ code: "custom", path: ["volumes", index, "floorCount"], message: `A volume may not extend past floor ${MAX_BUILDING_FLOORS}.` });
    });
    // Mesh-complexity guard: total stacked floors across volumes.
    const total = spec.volumes.reduce((sum, volume) => sum + volume.floorCount, 0);
    if (total > MAX_BUILDING_FLOORS * 2) ctx.addIssue({ code: "custom", path: ["volumes"], message: "Building is too complex to generate." });
  });
export type BuildingSpec = z.infer<typeof buildingSpecSchema>;

export const artifactSchema = z.object({
  id,
  kind: z.enum(["building-spec", "model-glb", "render-png"]),
  sourceNodeId: id,
  createdAt: isoDate,
  storageKey: z.string().min(1),
  metadata: z.record(z.string(), z.unknown()),
});
export type Artifact = z.infer<typeof artifactSchema>;

export const generationJobSchema = z.object({
  id,
  nodeId: id,
  provider: providerSchema,
  providerTaskId: z.string().optional(),
  status: z.enum(["queued", "running", "completed", "failed", "cancelled", "timed-out", "rate-limited"]),
  progress: z.number().min(0).max(100).optional(),
  outputUrl: z.string().url().optional(),
  error: z.object({ code: z.string(), message: z.string(), retryable: z.boolean() }).optional(),
  resultArtifactId: id.optional(),
  /** Execution timestamps; optional so jobs saved before P0.18 still validate. */
  createdAt: isoDate.optional(),
  updatedAt: isoDate.optional(),
  /** When the provider says its signed output URL / retained result expires, if known. */
  outputExpiresAt: isoDate.optional(),
});
export type GenerationJob = z.infer<typeof generationJobSchema>;

export const designNodeTypeSchema = z.enum(["prompt", "generation", "model", "render", "variation"]);
export type DesignNodeType = z.infer<typeof designNodeTypeSchema>;

export const designNodeSchema = z.object({
  id,
  type: designNodeTypeSchema,
  position: z.object({ x: finite, y: finite }),
  params: z.record(z.string(), z.unknown()),
  artifactId: id.optional(),
});
export type DesignNode = z.infer<typeof designNodeSchema>;

export const designEdgeSchema = z.object({
  id,
  source: id,
  sourcePort: z.string().min(1),
  target: id,
  targetPort: z.string().min(1),
});
export type DesignEdge = z.infer<typeof designEdgeSchema>;

export const designRevisionSchema = z.object({
  id,
  parentArtifactId: id,
  childArtifactId: id,
  sourceNodeIds: z.array(id),
  change: z.enum(["parameters", "prompt", "provider-regeneration"]),
  instruction: z.string().max(800),
  createdAt: isoDate,
});
export type DesignRevision = z.infer<typeof designRevisionSchema>;

export const viewerSettingsSchema = z.object({
  preset: z.enum(["perspective", "axonometric", "top", "front", "right"]),
  mode: z.enum(["shaded", "clay", "glass-concrete", "wireframe"]),
  grid: z.boolean(),
  axes: z.boolean(),
  shadows: z.boolean(),
});
export type ViewerSettings = z.infer<typeof viewerSettingsSchema>;

/** `viewer` is optional so projects saved before it existed still validate. */
export const projectSettingsSchema = z.object({ provider: providerSchema, viewer: viewerSettingsSchema.optional() });

export const siftProjectV2Schema = z
  .object({
    schemaVersion: z.literal(2),
    id,
    name: z.string().min(1).max(80),
    createdAt: isoDate,
    updatedAt: isoDate,
    viewport: z.object({ x: finite, y: finite, zoom: z.number().min(0.05).max(4) }),
    graph: z.object({ nodes: z.array(designNodeSchema), edges: z.array(designEdgeSchema) }),
    artifacts: z.record(z.string(), artifactSchema),
    jobs: z.record(z.string(), generationJobSchema),
    revisions: z.record(z.string(), designRevisionSchema),
    settings: projectSettingsSchema,
  })
  .superRefine((project, ctx) => {
    const nodeIds = new Set(project.graph.nodes.map((node) => node.id));
    if (nodeIds.size !== project.graph.nodes.length) ctx.addIssue({ code: "custom", path: ["graph", "nodes"], message: "Duplicate node ids." });
    project.graph.edges.forEach((edge, index) => {
      if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) ctx.addIssue({ code: "custom", path: ["graph", "edges", index], message: "Edge references a missing node." });
    });
    project.graph.nodes.forEach((node, index) => {
      if (node.artifactId && !(node.artifactId in project.artifacts)) ctx.addIssue({ code: "custom", path: ["graph", "nodes", index, "artifactId"], message: "Node references a missing artifact." });
    });
    Object.values(project.jobs).forEach((job) => {
      if (!nodeIds.has(job.nodeId)) ctx.addIssue({ code: "custom", path: ["jobs", job.id, "nodeId"], message: "Job references a missing node." });
      if (job.resultArtifactId && !(job.resultArtifactId in project.artifacts)) ctx.addIssue({ code: "custom", path: ["jobs", job.id, "resultArtifactId"], message: "Job references a missing artifact." });
    });
    Object.values(project.revisions).forEach((revision) => {
      if (!(revision.parentArtifactId in project.artifacts) || !(revision.childArtifactId in project.artifacts)) ctx.addIssue({ code: "custom", path: ["revisions", revision.id], message: "Revision references a missing artifact." });
    });
  });
export type SiftProjectV2 = z.infer<typeof siftProjectV2Schema>;

export const generateRequestSchema = z.object({
  prompt: z.string().trim().min(3).max(800),
  refinement: z.string().trim().max(400).default(""),
  provider: providerSchema,
  /** Must be literally `true` for paid (hosted) providers; the UI sets it only after the user confirms. */
  confirmSpend: z.boolean().optional(),
});


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

export const generateRequestSchema = z.object({
  prompt: z.string().trim().min(3).max(800),
  refinement: z.string().trim().max(400).default(""),
  provider: providerSchema,
});


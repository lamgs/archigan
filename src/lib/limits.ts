import type { BuildingSpec } from "./contracts";
import type { Layout } from "./geometry";

/** Hard budgets that keep a single building responsive on modest GPUs. */
export const MESH_LIMITS = {
  /** Triangles in the building geometry (ground plate excluded). */
  maxTriangles: 60_000,
  /** Separate draw calls (one per slab, two when glazed). */
  maxMeshes: 1_200,
  /** Triangles accepted when previewing an imported/hosted GLB; above this the model can only be downloaded. */
  maxHostedTriangles: 1_500_000,
} as const;

const BOX_TRIANGLES = 12;
const CYLINDER_SEGMENTS = 40; // keep in step with `three-building.ts`
const CYLINDER_TRIANGLES = CYLINDER_SEGMENTS * 4;

export type Complexity = { triangles: number; meshes: number; ok: boolean; message?: string };

/** Counts what `buildBuildingGroup` will create, without touching three.js. */
export function layoutComplexity(layout: Pick<Layout, "slabs">): Complexity {
  let triangles = 0;
  let meshes = 0;
  layout.slabs.forEach((slab) => {
    const solidHeight = slab.kind === "floor" ? slab.height * (1 - slab.glazing) : slab.height;
    const parts = (solidHeight > 0 ? 1 : 0) + (slab.kind === "floor" && slab.glazing > 0 ? 1 : 0);
    meshes += parts;
    triangles += parts * (slab.shape.type === "circle" ? CYLINDER_TRIANGLES : BOX_TRIANGLES);
  });
  if (triangles > MESH_LIMITS.maxTriangles || meshes > MESH_LIMITS.maxMeshes) {
    return { triangles, meshes, ok: false, message: `This building would need ${triangles.toLocaleString()} triangles in ${meshes.toLocaleString()} meshes, above the limit (${MESH_LIMITS.maxTriangles.toLocaleString()} / ${MESH_LIMITS.maxMeshes.toLocaleString()}). Reduce floors, volumes, or circular footprints.` };
  }
  return { triangles, meshes, ok: true };
}

/** Triangle count of a loaded three.js object graph (structural type so this stays free of three imports). */
export function countObjectTriangles(root: { traverse: (visit: (child: unknown) => void) => void }): number {
  let total = 0;
  root.traverse((child) => {
    const geometry = (child as { geometry?: { index?: { count: number } | null; attributes?: { position?: { count: number } } } }).geometry;
    if (!geometry) return;
    total += Math.floor((geometry.index?.count ?? geometry.attributes?.position?.count ?? 0) / 3);
  });
  return total;
}

export type SpecBudget = (spec: BuildingSpec) => Complexity;

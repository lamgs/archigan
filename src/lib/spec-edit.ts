import { buildingSpecSchema, type BuildingSpec } from "./contracts";
import { computeLayout } from "./geometry";
import { layoutComplexity } from "./limits";

type Volume = BuildingSpec["volumes"][number];
type VolumeNumberField = "startFloor" | "floorCount" | "footprintScale" | "offsetX" | "offsetZ" | "rotationDegrees" | "taper";

/** A single, serializable parameter change. Edits are data so they can be stored on nodes and replayed. */
export type SpecEdit =
  | { op: "floorHeight"; value: number }
  | { op: "footprint"; width?: number; depth?: number; radius?: number }
  | { op: "footprintType"; type: "rectangle" | "circle" }
  | { op: "facade"; style?: BuildingSpec["facade"]["style"]; glazingRatio?: number }
  | { op: "roof"; style: BuildingSpec["roof"]["style"] }
  | { op: "volume"; id: string; field: VolumeNumberField; value: number }
  | { op: "setback"; id: string; every: number | null; amount?: number }
  | { op: "material"; id: string; color?: string; kind?: BuildingSpec["materials"][string]["kind"] };

export type EditResult = { ok: true; spec: BuildingSpec } | { ok: false; message: string };

/** Ranges used by inspector inputs; the authoritative validation is still `buildingSpecSchema`. */
export const LIMITS = {
  floorHeight: { min: 2.5, max: 8, step: 0.1 },
  width: { min: 4, max: 200, step: 1 },
  depth: { min: 4, max: 200, step: 1 },
  radius: { min: 2, max: 100, step: 1 },
  glazingRatio: { min: 0, max: 1, step: 0.05 },
  startFloor: { min: 0, max: 120, step: 1 },
  floorCount: { min: 1, max: 120, step: 1 },
  footprintScale: { min: 0.1, max: 2, step: 0.05 },
  offsetX: { min: -80, max: 80, step: 1 },
  offsetZ: { min: -80, max: 80, step: 1 },
  rotationDegrees: { min: -180, max: 180, step: 5 },
  taper: { min: -0.5, max: 0.9, step: 0.05 },
  setbackEvery: { min: 1, max: 60, step: 1 },
  setbackAmount: { min: 0, max: 20, step: 0.1 },
} as const;

/** Applies an edit to a copy of `spec` and returns it only if the result is still a valid building. */
export function applyEdit(spec: BuildingSpec, edit: SpecEdit): EditResult {
  const next = structuredClone(spec);
  const volume = "id" in edit && edit.op !== "material" ? next.volumes.find((item) => item.id === edit.id) : undefined;
  switch (edit.op) {
    case "floorHeight":
      next.floorHeight = edit.value;
      break;
    case "footprint":
      if (next.footprint.type === "rectangle") {
        if (edit.width !== undefined) next.footprint.width = edit.width;
        if (edit.depth !== undefined) next.footprint.depth = edit.depth;
      } else if (edit.radius !== undefined) next.footprint.radius = edit.radius;
      break;
    case "footprintType":
      if (next.footprint.type !== edit.type) {
        next.footprint = next.footprint.type === "rectangle"
          ? { type: "circle", radius: Math.max(2, Math.min(100, Math.round(Math.sqrt((next.footprint.width * next.footprint.depth) / Math.PI)))) }
          : { type: "rectangle", width: Math.min(200, next.footprint.radius * 2), depth: Math.min(200, next.footprint.radius * 2) };
      }
      break;
    case "facade":
      if (edit.style) next.facade.style = edit.style;
      if (edit.glazingRatio !== undefined) next.facade.glazingRatio = edit.glazingRatio;
      break;
    case "roof":
      next.roof.style = edit.style;
      break;
    case "volume":
      if (!volume) return { ok: false, message: `Unknown volume "${edit.id}".` };
      volume[edit.field] = edit.value;
      break;
    case "setback":
      if (!volume) return { ok: false, message: `Unknown volume "${edit.id}".` };
      if (edit.every === null) {
        delete (volume as Partial<Volume>).setbackEvery;
        delete (volume as Partial<Volume>).setbackAmount;
      } else {
        volume.setbackEvery = edit.every;
        volume.setbackAmount = edit.amount ?? volume.setbackAmount ?? 1;
      }
      break;
    case "material": {
      const material = next.materials[edit.id];
      if (!material) return { ok: false, message: `Unknown material "${edit.id}".` };
      if (edit.color) material.color = edit.color;
      if (edit.kind) material.kind = edit.kind;
      break;
    }
  }
  const parsed = buildingSpecSchema.safeParse(next);
  if (parsed.success) {
    const budget = layoutComplexity(computeLayout(parsed.data));
    return budget.ok ? { ok: true, spec: parsed.data } : { ok: false, message: budget.message ?? "That change makes the building too complex." };
  }
  const issue = parsed.error.issues[0];
  const field = issue?.path.filter((part) => typeof part === "string").pop();
  return { ok: false, message: issue ? `${field ? `${String(field)}: ` : ""}${issue.message}` : "That change is not valid." };
}

/** Replays edits in order, skipping any that no longer apply (e.g. after the base spec changed). */
export function applyEdits(spec: BuildingSpec, edits: SpecEdit[]): BuildingSpec {
  return edits.reduce((current, edit) => {
    const result = applyEdit(current, edit);
    return result.ok ? result.spec : current;
  }, spec);
}

/** Key that identifies "the same control", so repeated edits of one field replace each other. */
export function editKey(edit: SpecEdit): string {
  switch (edit.op) {
    case "volume": return `volume:${edit.id}:${edit.field}`;
    case "setback": return `setback:${edit.id}`;
    case "material": return `material:${edit.id}:${edit.color ? "color" : "kind"}`;
    case "footprint": return `footprint:${edit.width !== undefined ? "w" : edit.depth !== undefined ? "d" : "r"}`;
    case "facade": return `facade:${edit.style ? "style" : "glazing"}`;
    default: return edit.op;
  }
}

export function mergeEdit(edits: SpecEdit[], edit: SpecEdit): SpecEdit[] {
  const key = editKey(edit);
  const kept = edits.filter((item) => editKey(item) !== key);
  // A footprint-type switch changes which footprint fields mean anything, so earlier size edits are dropped.
  return edit.op === "footprintType" ? [...kept.filter((item) => item.op !== "footprint"), edit] : [...kept, edit];
}

export function describeEdit(edit: SpecEdit): string {
  switch (edit.op) {
    case "floorHeight": return `Floor height → ${edit.value} m`;
    case "footprint": return `Footprint → ${[edit.width !== undefined && `width ${edit.width} m`, edit.depth !== undefined && `depth ${edit.depth} m`, edit.radius !== undefined && `radius ${edit.radius} m`].filter(Boolean).join(", ")}`;
    case "footprintType": return `Footprint shape → ${edit.type}`;
    case "facade": return `Facade → ${[edit.style, edit.glazingRatio !== undefined && `${Math.round(edit.glazingRatio * 100)}% glazing`].filter(Boolean).join(", ")}`;
    case "roof": return `Roof → ${edit.style}`;
    case "volume": return `${edit.id}: ${edit.field} → ${edit.value}`;
    case "setback": return edit.every === null ? `${edit.id}: setbacks removed` : `${edit.id}: setback ${edit.amount ?? ""} m every ${edit.every} floors`;
    case "material": return `Material ${edit.id} → ${[edit.color, edit.kind].filter(Boolean).join(", ")}`;
  }
}

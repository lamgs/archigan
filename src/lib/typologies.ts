import { buildingSpecSchema, type BuildingSpec } from "./contracts";
import { normalizeBrief } from "./massing";

export type Typology = "terraced" | "twin" | "cylindrical" | "rotated" | "low-rise";

type Params = { seed: number; material: MaterialName; floors: number; glass: boolean };
type MaterialName = "limestone" | "terracotta" | "concrete" | "glass";

const MATERIALS: Record<MaterialName, BuildingSpec["materials"][string]> = {
  limestone: { kind: "concrete", color: "#d8cfbb" },
  terracotta: { kind: "clay", color: "#b5543a" },
  concrete: { kind: "concrete", color: "#9a9a96" },
  glass: { kind: "glass", color: "#9cc4d4" },
};
const ACCENT = { kind: "metal" as const, color: "#4b4f52" };

function hash(value: string) {
  let h = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
const has = (text: string, terms: string[]) => terms.some((term) => text.includes(term));

export const normalizeBriefText = normalizeBrief;

export function detectTypology(brief: string): Typology {
  if (has(brief, ["twin", "two towers", "pair of towers", "paired"])) return "twin";
  if (has(brief, ["cylind", "round tower", "circular", "drum", "silo"])) return "cylindrical";
  if (has(brief, ["twist", "rotat", "spiral", "helical"])) return "rotated";
  if (has(brief, ["pavilion", "low-rise", "low rise", "archive", "single-story", "courtyard house"])) return "low-rise";
  return "terraced";
}

function materialOf(brief: string): MaterialName {
  if (has(brief, ["brick", "terracotta", "clay"])) return "terracotta";
  if (has(brief, ["glass", "transparent", "crystalline"])) return "glass";
  if (has(brief, ["concrete", "brutalist", "monolithic"])) return "concrete";
  return "limestone";
}

const vol = (id: string, role: BuildingSpec["volumes"][number]["role"], startFloor: number, floorCount: number, materialId: string, extra: Partial<BuildingSpec["volumes"][number]> = {}) => ({
  id, role, startFloor, floorCount, footprintScale: 1, offsetX: 0, offsetZ: 0, rotationDegrees: 0, taper: 0, materialId, ...extra,
});

/** Deterministically maps a brief to one of the supported typologies and its parameters. */
export function deriveBuildingSpec(prompt: string, refinement = ""): BuildingSpec {
  const brief = normalizeBrief(prompt, refinement);
  const seed = hash(brief);
  const typology = detectTypology(brief);
  const material = materialOf(brief);
  const tall = has(brief, ["tower", "high-rise", "skyscraper", "vertical"]);
  const stepped = has(brief, ["terrace", "stepped", "setback", "garden"]);
  const glass = material === "glass";
  const roofStyle = stepped || typology === "terraced" ? "terrace" : "flat";
  const p: Params = { seed, material, floors: tall ? 18 + (seed % 12) : 12 + (seed % 8), glass };
  const base = { schemaVersion: 1 as const, units: "m" as const, floorHeight: has(brief, ["gallery", "museum", "atrium"]) ? 4.6 : 3.6 };
  const facade = { style: glass ? ("grid" as const) : ("horizontal" as const), glazingRatio: glass ? 0.8 : 0.35 };
  const materials = { [material]: MATERIALS[material], accent: ACCENT };
  const name = prompt.trim().slice(0, 60) || "Untitled study";
  const twistDeg = 35 + (seed % 55);

  let spec: Omit<BuildingSpec, "schemaVersion" | "units" | "floorHeight" | "facade" | "materials" | "name">;
  switch (typology) {
    case "twin":
      spec = {
        footprint: { type: "rectangle", width: 64 + (seed % 10), depth: 34 + (seed % 6) },
        volumes: [
          vol("podium", "podium", 0, 4, "accent", { footprintScale: 1 }),
          vol("tower-a", "tower", 4, p.floors, material, { footprintScale: 0.34, offsetX: -17, taper: 0.1 }),
          vol("tower-b", "tower", 4, Math.max(6, p.floors - 5), material, { footprintScale: 0.34, offsetX: 17, taper: 0.1 }),
        ],
        roof: { style: "crown" },
      };
      break;
    case "cylindrical":
      spec = {
        footprint: { type: "circle", radius: 15 + (seed % 5) },
        volumes: [
          vol("podium", "podium", 0, 3, "accent", { footprintScale: 1.4 }),
          vol("tower", "tower", 3, p.floors + 4, material, { taper: 0.25, rotationDegrees: 0 }),
        ],
        roof: { style: "crown" },
      };
      break;
    case "rotated":
      spec = {
        footprint: { type: "rectangle", width: 24 + (seed % 8), depth: 24 + ((seed >>> 3) % 8) },
        volumes: [vol("tower", "tower", 0, p.floors + 4, material, { rotationDegrees: twistDeg, taper: 0.12 })],
        roof: { style: "flat" },
      };
      break;
    case "low-rise":
      spec = {
        footprint: { type: "rectangle", width: 56 + (seed % 14), depth: 22 + (seed % 8) },
        volumes: [vol("main", "wing", 0, 3 + (seed % 2), material), vol("wing", "wing", 0, 2, "accent", { footprintScale: 0.5, offsetZ: 20 })],
        roof: { style: "flat" },
      };
      break;
    default:
      spec = {
        footprint: { type: "rectangle", width: 56 + (seed % 10), depth: 38 + (seed % 8) },
        volumes: [
          vol("podium", "podium", 0, 4, "accent"),
          vol("tower", "tower", 4, p.floors, material, { footprintScale: 0.6, offsetZ: -4, setbackEvery: 4, setbackAmount: 1.2 + (seed % 5) / 5 }),
        ],
        roof: { style: roofStyle },
      };
  }
  return buildingSpecSchema.parse({ ...base, name, facade, materials, ...spec });
}

export function describeSpec(spec: BuildingSpec) {
  const levels = Math.max(...spec.volumes.map((volume) => volume.startFloor + volume.floorCount));
  const footprint = spec.footprint.type === "circle" ? `⌀ ${Math.round(spec.footprint.radius * 2)} m` : `${Math.round(spec.footprint.width)} × ${Math.round(spec.footprint.depth)} m`;
  return { levels, volumes: spec.volumes.length, footprint };
}

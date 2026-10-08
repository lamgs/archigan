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

const NUMBER_WORDS: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10, eleven: 11, twelve: 12, thirteen: 13, fourteen: 14, fifteen: 15, sixteen: 16, seventeen: 17, eighteen: 18, nineteen: 19, twenty: 20, thirty: 30, forty: 40 };
const toCount = (token: string) => (/^\d+$/.test(token) ? Number(token) : NUMBER_WORDS[token]);
const COUNT = "(\\d+|[a-z]+)";
const STOREYS = "(?:stor(?:y|ies|ey|eys)|floors?|levels?)";

/** Floor counts and setback rhythm the local interpreter can read from a brief ("12-story", "four-story podium", "eight-story tower", "setbacks every two floors"). */
export type ParsedCounts = { total?: number; podium?: number; tower?: number; setbackEvery?: number };

export function parseCounts(brief: string): ParsedCounts {
  const out: ParsedCounts = {};
  const roleSpans: [number, number][] = [];
  for (const match of brief.matchAll(new RegExp(`${COUNT}[- ]${STOREYS}(?:[ -][a-z]+){0,3}?[ -](podium|base|tower)\\b`, "g"))) {
    const n = toCount(match[1]);
    if (!n) continue;
    out[match[2] === "tower" ? "tower" : "podium"] ??= n;
    roleSpans.push([match.index ?? 0, (match.index ?? 0) + match[0].length]);
  }
  for (const match of brief.matchAll(new RegExp(`${COUNT}[- ]${STOREYS}\\b`, "g"))) {
    const start = match.index ?? 0;
    const n = toCount(match[1]);
    const afterEvery = brief.slice(Math.max(0, start - 6), start) === "every ";
    if (n && !afterEvery && !roleSpans.some(([a, b]) => start >= a && start < b) && !out.total) out.total = n;
  }
  // "a 20-story terraced tower" describes the whole building; "an eight-story tower above a podium" describes the tower part.
  if (out.tower !== undefined && out.podium === undefined && out.total === undefined && !/\b(above|atop|on top|over|rising from)\b/.test(brief)) {
    out.total = out.tower;
    delete out.tower;
  }
  const every = new RegExp(`every ${COUNT} ${STOREYS}\\b`).exec(brief);
  if (every && toCount(every[1])) out.setbackEvery = toCount(every[1]);
  return out;
}

/** What the local (offline) interpreter understands. Everything else in a brief is ignored, and the UI says so. */
export const INTERPRETER_HELP = "The local engine reads these keywords: shape — twin, cylindrical/round, twist/spiral, pavilion/low-rise (otherwise a terraced tower); material — glass/glazed, brick/terracotta, concrete (otherwise limestone); form — tower/skyscraper (taller), terrace/stepped/setback/garden (stepped roof), gallery/museum/atrium (taller floors); numbers — “12-story”, “four-story podium”, “eight-story tower”, “setbacks every two floors”. Other words are ignored.";

export function detectTypology(brief: string): Typology {
  if (has(brief, ["twin", "two towers", "pair of towers", "paired"])) return "twin";
  if (has(brief, ["cylind", "round tower", "circular", "drum", "silo"])) return "cylindrical";
  if (has(brief, ["twist", "rotat", "spiral", "helical"])) return "rotated";
  if (has(brief, ["pavilion", "low-rise", "low rise", "archive", "single-story", "courtyard house"])) return "low-rise";
  return "terraced";
}

function materialOf(brief: string): MaterialName {
  if (has(brief, ["brick", "terracotta", "clay"])) return "terracotta";
  if (has(brief, ["glass", "glazed", "glazing", "transparent", "crystalline"])) return "glass";
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
  const counts = parseCounts(brief);
  const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, Math.round(value)));
  const podiumFloors = clamp(counts.podium ?? 4, 1, 12);
  const towerFloors = clamp(counts.tower ?? (counts.total !== undefined ? counts.total - podiumFloors : p.floors), 1, 100);
  const setbackEvery = clamp(counts.setbackEvery ?? 4, 1, 60);
  const totalFloors = clamp(counts.total ?? p.floors + 4, 2, 110);

  let spec: Omit<BuildingSpec, "schemaVersion" | "units" | "floorHeight" | "facade" | "materials" | "name">;
  switch (typology) {
    case "twin":
      spec = {
        footprint: { type: "rectangle", width: 64 + (seed % 10), depth: 34 + (seed % 6) },
        volumes: [
          vol("podium", "podium", 0, podiumFloors, "accent", { footprintScale: 1 }),
          vol("tower-a", "tower", podiumFloors, counts.tower !== undefined || counts.total !== undefined ? towerFloors : p.floors, material, { footprintScale: 0.34, offsetX: -17, taper: 0.1 }),
          vol("tower-b", "tower", podiumFloors, Math.max(3, (counts.tower !== undefined || counts.total !== undefined ? towerFloors : p.floors) - 5), material, { footprintScale: 0.34, offsetX: 17, taper: 0.1 }),
        ],
        roof: { style: "crown" },
      };
      break;
    case "cylindrical":
      spec = {
        footprint: { type: "circle", radius: 15 + (seed % 5) },
        volumes: [
          vol("podium", "podium", 0, 3, "accent", { footprintScale: 1.4 }),
          vol("tower", "tower", 3, counts.total !== undefined ? Math.max(1, totalFloors - 3) : p.floors + 4, material, { taper: 0.25, rotationDegrees: 0 }),
        ],
        roof: { style: "crown" },
      };
      break;
    case "rotated":
      spec = {
        footprint: { type: "rectangle", width: 24 + (seed % 8), depth: 24 + ((seed >>> 3) % 8) },
        volumes: [vol("tower", "tower", 0, counts.total !== undefined ? totalFloors : p.floors + 4, material, { rotationDegrees: twistDeg, taper: 0.12 })],
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
          vol("podium", "podium", 0, podiumFloors, "accent"),
          vol("tower", "tower", podiumFloors, counts.tower !== undefined || counts.total !== undefined ? towerFloors : p.floors, material, { footprintScale: 0.6, offsetZ: -4, setbackEvery, setbackAmount: 1.2 + (seed % 5) / 5 }),
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

import type { BuildingSpec } from "./contracts";

export type SlabShape = { type: "rectangle"; width: number; depth: number } | { type: "circle"; radius: number };

export type Slab = {
  volumeId: string;
  kind: "floor" | "roof";
  level: number;
  shape: SlabShape;
  /** Centre of the slab in metres (Y is the slab's base). */
  x: number;
  y: number;
  z: number;
  height: number;
  rotationY: number;
  materialId: string;
  /** Fraction of a floor's height that is glazed (0 for solid facades and roofs). */
  glazing: number;
};

export type Layout = {
  slabs: Slab[];
  bounds: { min: [number, number, number]; max: [number, number, number] };
  floors: number;
  warnings: string[];
};

const FLOOR_FILL = 0.92; // leaves a visible reveal between floors
const MIN_RECT = 2;
const MIN_RADIUS = 1;
const CROWN_HEIGHT = 1.2;
const TERRACE_HEIGHT = 0.6;
const TERRACE_INSET = 1.2;

/** Pure, deterministic expansion of a BuildingSpec into per-floor slabs. No rendering dependencies. */
export function computeLayout(spec: BuildingSpec): Layout {
  const slabs: Slab[] = [];
  const warnings = new Set<string>();
  const glazing = spec.facade.style === "solid" ? 0 : spec.facade.glazingRatio;

  spec.volumes.forEach((volume) => {
    const baseShape = (scale: number, taper: number, inset: number): SlabShape => {
      if (spec.footprint.type === "circle") {
        const radius = spec.footprint.radius * scale * taper - inset;
        if (radius < MIN_RADIUS) warnings.add(`Volume "${volume.id}" shrinks below the minimum size; clamped.`);
        return { type: "circle", radius: Math.max(MIN_RADIUS, radius) };
      }
      const width = spec.footprint.width * scale * taper - inset * 2;
      const depth = spec.footprint.depth * scale * taper - inset * 2;
      if (width < MIN_RECT || depth < MIN_RECT) warnings.add(`Volume "${volume.id}" shrinks below the minimum size; clamped.`);
      return { type: "rectangle", width: Math.max(MIN_RECT, width), depth: Math.max(MIN_RECT, depth) };
    };

    let top: Slab | undefined;
    for (let index = 0; index < volume.floorCount; index += 1) {
      const progress = volume.floorCount > 1 ? index / (volume.floorCount - 1) : 0;
      const steps = volume.setbackEvery ? Math.floor(index / volume.setbackEvery) : 0;
      const inset = (volume.setbackAmount ?? 0) * steps;
      const level = volume.startFloor + index;
      top = {
        volumeId: volume.id,
        kind: "floor",
        level,
        shape: baseShape(volume.footprintScale, 1 - volume.taper * progress, inset),
        x: volume.offsetX,
        y: level * spec.floorHeight,
        z: volume.offsetZ,
        height: spec.floorHeight * FLOOR_FILL,
        // rotationDegrees is the total twist from the first to the last floor of the volume.
        rotationY: (volume.rotationDegrees * progress * Math.PI) / 180,
        materialId: volume.materialId,
        glazing,
      };
      slabs.push(top);
    }

    if (top && spec.roof.style !== "flat") {
      const crown = spec.roof.style === "crown";
      const shape: SlabShape =
        top.shape.type === "circle"
          ? { type: "circle", radius: Math.max(MIN_RADIUS, top.shape.radius + (crown ? 0.5 : -TERRACE_INSET)) }
          : { type: "rectangle", width: Math.max(MIN_RECT, top.shape.width + (crown ? 1 : -TERRACE_INSET * 2)), depth: Math.max(MIN_RECT, top.shape.depth + (crown ? 1 : -TERRACE_INSET * 2)) };
      slabs.push({ ...top, kind: "roof", level: top.level + 1, shape, y: top.y + top.height, height: crown ? CROWN_HEIGHT : TERRACE_HEIGHT, glazing: 0 });
    }
  });

  return { slabs, bounds: boundsOf(slabs), floors: spec.volumes.reduce((sum, volume) => sum + volume.floorCount, 0), warnings: [...warnings] };
}

function boundsOf(slabs: Slab[]): Layout["bounds"] {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  slabs.forEach((slab) => {
    const corners: [number, number][] =
      slab.shape.type === "circle"
        ? [[slab.shape.radius, slab.shape.radius], [-slab.shape.radius, slab.shape.radius], [slab.shape.radius, -slab.shape.radius], [-slab.shape.radius, -slab.shape.radius]]
        : rotatedCorners(slab.shape.width / 2, slab.shape.depth / 2, slab.rotationY);
    corners.forEach(([cx, cz]) => {
      min[0] = Math.min(min[0], slab.x + cx);
      max[0] = Math.max(max[0], slab.x + cx);
      min[2] = Math.min(min[2], slab.z + cz);
      max[2] = Math.max(max[2], slab.z + cz);
    });
    min[1] = Math.min(min[1], slab.y);
    max[1] = Math.max(max[1], slab.y + slab.height);
  });
  return slabs.length ? { min, max } : { min: [0, 0, 0], max: [0, 0, 0] };
}

function rotatedCorners(hw: number, hd: number, angle: number): [number, number][] {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return ([[hw, hd], [-hw, hd], [hw, -hd], [-hw, -hd]] as [number, number][]).map(([x, z]) => [x * cos + z * sin, -x * sin + z * cos]);
}

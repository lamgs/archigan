import type { Layout } from "./geometry";

export const CAMERA_PRESETS = ["perspective", "axonometric", "top", "front", "right"] as const;
export type CameraPreset = (typeof CAMERA_PRESETS)[number];
export const VIEW_MODES = ["shaded", "clay", "glass-concrete", "wireframe"] as const;
export type ViewMode = (typeof VIEW_MODES)[number];

/** Longest side of the model after scaling, in scene units. Must match `three-building.ts`. */
export const TARGET_SIZE = 28;
export const PERSPECTIVE_FOV = 38;

export type SceneMetrics = {
  scale: number;
  /** Scaled model size [x, y, z]. */
  size: [number, number, number];
  /** Scaled centre of the model; X/Z are 0 because the builder centres the group. */
  center: [number, number, number];
  /** Radius of the sphere enclosing the scaled model. */
  radius: number;
};

export function sceneMetrics(layout: Pick<Layout, "bounds">): SceneMetrics {
  const raw = layout.bounds.max.map((value, axis) => value - layout.bounds.min[axis]);
  const scale = TARGET_SIZE / Math.max(1, ...raw);
  const size = raw.map((value) => value * scale) as [number, number, number];
  const radius = Math.max(0.5, Math.hypot(size[0], size[1], size[2]) / 2);
  return { scale, size, center: [0, layout.bounds.min[1] * scale + size[1] / 2, 0], radius };
}

/** Distance at which a sphere of `radius` fits inside a perspective frustum (limited by the narrower axis). */
export function framingDistance(radius: number, fovDegrees: number, aspect: number, margin = 1.15) {
  const vertical = (fovDegrees * Math.PI) / 180;
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * Math.max(0.1, aspect));
  const limiting = Math.min(vertical, horizontal);
  return (radius * margin) / Math.sin(limiting / 2);
}

export type Pose = {
  projection: "perspective" | "orthographic";
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  /** Orthographic only: pixels per scene unit so the model fills the viewport with a margin. */
  zoom?: number;
  /** Whether free orbiting makes sense for this preset. */
  orbit: boolean;
};

const normalize = (v: [number, number, number]): [number, number, number] => {
  const length = Math.hypot(...v) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
};

export function presetPose(preset: CameraPreset, metrics: SceneMetrics, viewport: { width: number; height: number }): Pose {
  const aspect = viewport.width / Math.max(1, viewport.height);
  const [cx, cy, cz] = metrics.center;
  const target: [number, number, number] = [cx, cy, cz];
  const far = metrics.radius * 4; // orthographic cameras only need to sit outside the model
  const fit = (extentX: number, extentY: number, margin = 1.2) => Math.max(1, Math.min(viewport.width / (extentX * margin), viewport.height / (extentY * margin)));
  const [sx, sy, sz] = metrics.size;
  const at = (dir: [number, number, number], distance: number): [number, number, number] => {
    const n = normalize(dir);
    return [cx + n[0] * distance, cy + n[1] * distance, cz + n[2] * distance];
  };
  switch (preset) {
    case "top":
      return { projection: "orthographic", position: at([0, 1, 0], far), target, up: [0, 0, -1], zoom: fit(sx, sz), orbit: false };
    case "front":
      return { projection: "orthographic", position: at([0, 0, 1], far), target, up: [0, 1, 0], zoom: fit(sx, sy), orbit: false };
    case "right":
      return { projection: "orthographic", position: at([1, 0, 0], far), target, up: [0, 1, 0], zoom: fit(sz, sy), orbit: false };
    case "axonometric":
      return { projection: "orthographic", position: at([1, 0.82, 1], far), target, up: [0, 1, 0], zoom: fit(metrics.radius * 2, metrics.radius * 2, 1.1), orbit: true };
    default:
      return { projection: "perspective", position: at([1, 0.75, 1], framingDistance(metrics.radius, PERSPECTIVE_FOV, aspect)), target, up: [0, 1, 0], orbit: true };
  }
}

export const PRESET_LABELS: Record<CameraPreset, string> = { perspective: "Perspective", axonometric: "Axonometric", top: "Top", front: "Front", right: "Right" };
export const MODE_LABELS: Record<ViewMode, string> = { shaded: "Shaded", clay: "Clay", "glass-concrete": "Glass + concrete", wireframe: "Wireframe" };

import type { BuildingSpec } from "./contracts";
import { CAMERA_PRESETS, VIEW_MODES, type CameraPreset, type ViewMode } from "./viewer";

export const LIGHTING = ["studio", "soft", "dusk"] as const;
export type Lighting = (typeof LIGHTING)[number];
export const BACKGROUNDS = ["paper", "white", "charcoal", "transparent"] as const;
export type Background = (typeof BACKGROUNDS)[number];

export const RESOLUTIONS = {
  "1024x1024": { width: 1024, height: 1024, label: "1024 × 1024 (square)" },
  "1600x900": { width: 1600, height: 900, label: "1600 × 900 (wide)" },
  "1920x1080": { width: 1920, height: 1080, label: "1920 × 1080 (full HD)" },
} as const;
export type ResolutionId = keyof typeof RESOLUTIONS;

export type RenderSettings = { preset: CameraPreset; mode: ViewMode; lighting: Lighting; background: Background; resolution: ResolutionId };

export const DEFAULT_RENDER_SETTINGS: RenderSettings = { preset: "axonometric", mode: "shaded", lighting: "studio", background: "paper", resolution: "1600x900" };

export const BACKGROUND_COLORS: Record<Exclude<Background, "transparent">, string> = { paper: "#d9d3c6", white: "#ffffff", charcoal: "#2b2a28" };

export type GpuLimits = { maxRenderbufferSize: number; maxViewportDims: [number, number]; deviceMemoryGb?: number };

const pick = <T extends string>(value: unknown, options: readonly T[], fallback: T): T => (options.includes(value as T) ? (value as T) : fallback);

/** Reads render settings from a node's free-form params, replacing anything missing or invalid with defaults. */
export function parseRenderSettings(params: Record<string, unknown>): RenderSettings {
  const d = DEFAULT_RENDER_SETTINGS;
  return {
    preset: pick(params.preset, CAMERA_PRESETS, d.preset),
    mode: pick(params.mode, VIEW_MODES, d.mode),
    lighting: pick(params.lighting, LIGHTING, d.lighting),
    background: pick(params.background, BACKGROUNDS, d.background),
    resolution: pick(params.resolution, Object.keys(RESOLUTIONS) as ResolutionId[], d.resolution),
  };
}

/**
 * Resolutions this device can render. The two baseline sizes are offered whenever the GPU allows them; full HD is
 * additionally withheld on devices that report little memory, where the offscreen buffer is likely to fail.
 */
export function supportedResolutions(limits: GpuLimits | null): ResolutionId[] {
  if (!limits) return [];
  const fits = (id: ResolutionId) => {
    const { width, height } = RESOLUTIONS[id];
    return limits.maxRenderbufferSize >= Math.max(width, height) && limits.maxViewportDims[0] >= width && limits.maxViewportDims[1] >= height;
  };
  return (Object.keys(RESOLUTIONS) as ResolutionId[]).filter((id) => fits(id) && (id !== "1920x1080" || limits.deviceMemoryGb === undefined || limits.deviceMemoryGb >= 4));
}

function hash(text: string) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(16);
}

/** Identifies exactly what a render depends on, so a render can be recognised as up to date or out of date. */
export function renderInputKey(spec: BuildingSpec, settings: RenderSettings) {
  return hash(JSON.stringify({ spec, settings }));
}

export function describeRender(settings: RenderSettings) {
  const { width, height } = RESOLUTIONS[settings.resolution];
  return `${width}×${height} · ${settings.preset} · ${settings.mode}`;
}

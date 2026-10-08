import * as THREE from "three";
import type { BuildingSpec } from "./contracts";
import { computeLayout } from "./geometry";
import { BACKGROUND_COLORS, RESOLUTIONS, type GpuLimits, type RenderSettings } from "./render-settings";
import { buildBuildingGroup, disposeBuildingGroup } from "./three-building";
import { PERSPECTIVE_FOV, presetPose, sceneMetrics } from "./viewer";

export class RenderError extends Error {}

/** Probes the GPU once; returns null when WebGL is unavailable. */
export function probeGpu(): GpuLimits | null {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
    if (!gl) return null;
    const dims = gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array;
    const limits: GpuLimits = {
      maxRenderbufferSize: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number,
      maxViewportDims: [dims[0], dims[1]],
      deviceMemoryGb: (navigator as Navigator & { deviceMemory?: number }).deviceMemory,
    };
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return limits;
  } catch {
    return null;
  }
}

function addLights(scene: THREE.Scene, lighting: RenderSettings["lighting"], metrics: ReturnType<typeof sceneMetrics>) {
  const spread = Math.max(34, metrics.radius * 1.4);
  const sun = (color: string, intensity: number, position: [number, number, number], shadows: boolean) => {
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(...position);
    light.castShadow = shadows;
    if (shadows) {
      light.shadow.mapSize.set(2048, 2048);
      Object.assign(light.shadow.camera, { left: -spread, right: spread, top: spread, bottom: -spread, near: 1, far: 160 });
      light.shadow.bias = -0.0004;
    }
    scene.add(light);
  };
  if (lighting === "soft") {
    scene.add(new THREE.AmbientLight("#ffffff", 1.9));
    sun("#ffffff", 0.9, [10, 30, 14], false);
  } else if (lighting === "dusk") {
    scene.add(new THREE.AmbientLight("#8aa0c8", 0.7));
    sun("#ffb98a", 2.8, [34, 12, 10], true);
  } else {
    scene.add(new THREE.AmbientLight("#ffffff", 1.1));
    sun("#ffffff", 2.2, [18, 34, 20], true);
  }
}

/** Renders the model offscreen at exactly the requested pixel size and returns a PNG. */
export async function renderPng(spec: BuildingSpec, settings: RenderSettings): Promise<{ blob: Blob; width: number; height: number }> {
  const { width, height } = RESOLUTIONS[settings.resolution];
  const canvas = document.createElement("canvas");
  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: settings.background === "transparent", preserveDrawingBuffer: true });
  } catch {
    throw new RenderError("WebGL is not available, so the render could not be created.");
  }
  const layout = computeLayout(spec);
  const group = buildBuildingGroup(spec, layout, settings.mode);
  try {
    renderer.toneMapping = THREE.ACESFilmicToneMapping; // match the live viewer (react-three-fiber defaults to ACES Filmic)
    renderer.toneMappingExposure = 1;
    renderer.setPixelRatio(1);
    renderer.setSize(width, height, false);
    if (canvas.width !== width || canvas.height !== height) throw new RenderError(`This device could not allocate a ${width}×${height} image.`);
    renderer.shadowMap.enabled = settings.lighting !== "soft";
    renderer.shadowMap.type = THREE.PCFShadowMap;
    const scene = new THREE.Scene();
    if (settings.background !== "transparent") scene.background = new THREE.Color(BACKGROUND_COLORS[settings.background]);
    const metrics = sceneMetrics(layout);
    addLights(scene, settings.lighting, metrics);
    scene.add(group);

    const pose = presetPose(settings.preset, metrics, { width, height });
    const camera = pose.projection === "orthographic"
      ? new THREE.OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, 0.1, metrics.radius * 20)
      : new THREE.PerspectiveCamera(PERSPECTIVE_FOV, width / height, 0.1, metrics.radius * 40);
    if (camera instanceof THREE.OrthographicCamera && pose.zoom) camera.zoom = pose.zoom;
    camera.up.set(...pose.up);
    camera.position.set(...pose.position);
    camera.lookAt(...pose.target);
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!blob) throw new RenderError("The browser could not encode the PNG.");
    return { blob, width, height };
  } finally {
    disposeBuildingGroup(group);
    renderer.dispose();
    renderer.forceContextLoss();
  }
}

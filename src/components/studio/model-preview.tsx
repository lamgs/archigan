"use client";

import { providerLabel } from "@/lib/provider-meta";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box3, MathUtils, Object3D, OrthographicCamera, PerspectiveCamera, Spherical, Vector3, type Group } from "three";
import type { OrbitControls as OrbitControlsType } from "three/examples/jsm/controls/OrbitControls.js";
import type { BuildingSpec, Provider, ViewerSettings } from "@/lib/contracts";
import { computeLayout } from "@/lib/geometry";
import { countObjectTriangles, layoutComplexity, MESH_LIMITS } from "@/lib/limits";
import { probeGpu } from "@/lib/render-image";
import { buildBuildingGroup, disposeBuildingGroup } from "@/lib/three-building";
import { describeSpec } from "@/lib/typologies";
import { ViewerBoundary, ViewerFallback } from "./viewer-fallback";
import { CAMERA_PRESETS, DEFAULT_VIEWER_SETTINGS, MODE_LABELS, PERSPECTIVE_FOV, PRESET_LABELS, VIEW_MODES, metricsFromSize, presetPose, sceneMetrics, type CameraPreset, type SceneMetrics, type ViewMode } from "@/lib/viewer";

export type HostedPreview = { blob: Blob; label: string };
type LoadedHosted = { blob: Blob; object: Object3D; metrics: SceneMetrics };

type RigHandle = { frame: () => void; orbit: (azimuth: number, polar: number) => void; zoom: (factor: number) => void };

function CameraRig({ preset, metrics, frameToken, onRig }: { preset: CameraPreset; metrics: SceneMetrics; frameToken: number; onRig: (handle: RigHandle) => void }) {
  const { gl, size, set, invalidate } = useThree();
  const controls = useRef<OrbitControlsType | null>(null);
  const cameraRef = useRef<PerspectiveCamera | OrthographicCamera | null>(null);
  const sizeRef = useRef(size);
  useEffect(() => { sizeRef.current = size; }, [size]);

  // Cameras we create are not managed by R3F, so keep their frustum/aspect in sync with the viewport ourselves.
  const syncProjection = useCallback((camera: PerspectiveCamera | OrthographicCamera, viewport: { width: number; height: number }) => {
    if (camera instanceof OrthographicCamera) {
      camera.left = -viewport.width / 2;
      camera.right = viewport.width / 2;
      camera.top = viewport.height / 2;
      camera.bottom = -viewport.height / 2;
    } else camera.aspect = viewport.width / Math.max(1, viewport.height);
    camera.updateProjectionMatrix();
  }, []);

  const applyPose = useCallback(() => {
    const camera = cameraRef.current;
    if (!camera) return;
    const pose = presetPose(preset, metrics, sizeRef.current);
    syncProjection(camera, sizeRef.current);
    camera.position.set(...pose.position);
    camera.up.set(...pose.up);
    if (camera instanceof OrthographicCamera && pose.zoom) camera.zoom = pose.zoom;
    camera.lookAt(...pose.target);
    camera.updateProjectionMatrix();
    controls.current?.target.set(...pose.target);
    controls.current?.update();
    invalidate();
  }, [preset, metrics, invalidate, syncProjection]);

  // (Re)create the camera and controls when the projection type changes.
  useEffect(() => {
    let disposed = false;
    const pose = presetPose(preset, metrics, sizeRef.current);
    const camera = pose.projection === "orthographic"
      ? new OrthographicCamera(-1, 1, 1, -1, 0.1, metrics.radius * 20)
      : new PerspectiveCamera(PERSPECTIVE_FOV, sizeRef.current.width / Math.max(1, sizeRef.current.height), 0.1, metrics.radius * 40);
    // OrbitControls caches camera.up at construction, so the pose is fixed before the controls exist.
    camera.up.set(...pose.up);
    camera.position.set(...pose.position);
    camera.lookAt(...pose.target);
    cameraRef.current = camera;
    set({ camera });
    void import("three/examples/jsm/controls/OrbitControls.js").then(({ OrbitControls }) => {
      if (disposed) return;
      const instance = new OrbitControls(camera, gl.domElement);
      instance.enableDamping = true;
      instance.dampingFactor = 0.08;
      instance.maxPolarAngle = Math.PI * 0.495;
      instance.enableRotate = pose.orbit;
      instance.addEventListener("change", () => invalidate());
      controls.current = instance;
      applyPose();
    });
    return () => {
      disposed = true;
      controls.current?.dispose();
      controls.current = null;
    };
    // Rebuilt per preset (not just per projection kind) because OrbitControls cannot follow a changing `up` vector.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, gl, set]);

  useEffect(() => {
    if (cameraRef.current) { syncProjection(cameraRef.current, size); invalidate(); }
  }, [size, syncProjection, invalidate]);

  useEffect(() => {
    if (controls.current) controls.current.enableRotate = presetPose(preset, metrics, size).orbit;
    applyPose();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset, frameToken, Math.round(metrics.radius * 10), applyPose]);

  useEffect(() => {
    onRig({
      frame: applyPose,
      orbit: (azimuth, polar) => {
        const camera = cameraRef.current;
        const target = controls.current?.target;
        if (!camera || !target || !presetPose(preset, metrics, sizeRef.current).orbit) return;
        const offset = new Vector3().copy(camera.position).sub(target);
        const spherical = new Spherical().setFromVector3(offset);
        spherical.theta += azimuth;
        spherical.phi = MathUtils.clamp(spherical.phi + polar, 0.05, Math.PI * 0.495);
        camera.position.copy(target).add(offset.setFromSpherical(spherical));
        camera.lookAt(target);
        controls.current?.update();
        invalidate();
      },
      zoom: (factor) => {
        const camera = cameraRef.current;
        const target = controls.current?.target;
        if (!camera || !target) return;
        if (camera instanceof OrthographicCamera) camera.zoom = MathUtils.clamp(camera.zoom * factor, 1, 2000);
        else camera.position.sub(target).multiplyScalar(1 / factor).add(target);
        camera.updateProjectionMatrix();
        controls.current?.update();
        invalidate();
      },
    });
  });

  useFrame(() => controls.current?.update());
  return null;
}

function Model({ spec, mode }: { spec: BuildingSpec; mode: ViewMode }) {
  const group = useMemo(() => buildBuildingGroup(spec, undefined, mode), [spec, mode]);
  useEffect(() => () => disposeBuildingGroup(group), [group]);
  return <primitive object={group} />;
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
}

/** Lightweight GPU diagnostics (counts only) used by automated leak checks; exposes no model data. */
function exposeDiagnostics(gl: { info: { memory: { geometries: number; textures: number }; programs?: unknown[] | null; render: { frame: number } } }) {
  (window as unknown as { __siftGpu?: () => { geometries: number; textures: number; programs: number; frames: number } }).__siftGpu = () => ({ geometries: gl.info.memory.geometries, textures: gl.info.memory.textures, programs: gl.info.programs?.length ?? 0, frames: gl.info.render.frame });
}

function Toggle({ label, pressed, onChange, shortcut }: { label: string; pressed: boolean; onChange: (value: boolean) => void; shortcut?: string }) {
  return <button type="button" aria-pressed={pressed} aria-keyshortcuts={shortcut} className="viewer-toggle" onClick={() => onChange(!pressed)}>{label}</button>;
}

export function ModelPreview({ spec: specProp, hosted, provider, stale = false, settings = DEFAULT_VIEWER_SETTINGS, onSettings }: { spec?: BuildingSpec; hosted?: HostedPreview; provider: Provider; stale?: boolean; settings?: ViewerSettings; onSettings?: (settings: ViewerSettings) => void }) {
  // Callers recompute specs as fresh objects on unrelated state changes (autosave, selection); key by content so the
  // model is only rebuilt (and GPU buffers re-uploaded) when the geometry actually changes.
  const specKey = specProp ? JSON.stringify(specProp) : "";
  const spec = useMemo(() => (specKey ? (JSON.parse(specKey) as BuildingSpec) : undefined), [specKey]);
  const canvasWrap = useRef<HTMLDivElement>(null);
  const focusButton = useRef<HTMLButtonElement>(null);
  const rig = useRef<RigHandle | null>(null);
  const onRig = useCallback((handle: RigHandle) => { rig.current = handle; }, []);
  const { preset, mode, grid, axes, shadows } = settings;
  const update = (patch: Partial<ViewerSettings>) => onSettings?.({ ...settings, ...patch });
  const setPreset = (value: CameraPreset) => update({ preset: value });
  const setMode = (value: ViewMode) => update({ mode: value });
  const setGrid = (value: boolean) => update({ grid: value });
  const setAxes = (value: boolean) => update({ axes: value });
  const setShadows = (value: boolean) => update({ shadows: value });
  const [focus, setFocus] = useState(false);
  const [frameToken, setFrameToken] = useState(0);
  const [exportState, setExportState] = useState<"idle" | "exporting" | "complete" | "error">("idle");
  const [webglSupported] = useState(() => probeGpu() !== null);
  const [contextLost, setContextLost] = useState(false);
  const [canvasKey, setCanvasKey] = useState(0);
  const [pngError, setPngError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<LoadedHosted | null>(null);
  const [hostedError, setHostedError] = useState<string | null>(null);
  const hostedBlob = hosted?.blob;
  useEffect(() => {
    if (!hostedBlob) return;
    let cancelled = false;
    void (async () => {
      try {
        const { GLTFLoader } = await import("three/examples/jsm/loaders/GLTFLoader.js");
        const gltf = await new GLTFLoader().parseAsync(await hostedBlob.arrayBuffer(), "");
        if (cancelled) return;
        const box = new Box3().setFromObject(gltf.scene);
        const raw = box.getSize(new Vector3());
        if (!Number.isFinite(raw.x + raw.y + raw.z) || raw.x + raw.y + raw.z === 0) throw new Error("empty");
        if (countObjectTriangles(gltf.scene) > MESH_LIMITS.maxHostedTriangles) throw new Error("heavy");
        const metrics = metricsFromSize([raw.x, raw.y, raw.z]);
        const holder = new Object3D();
        gltf.scene.position.set(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
        holder.add(gltf.scene);
        holder.scale.setScalar(metrics.scale);
        setLoaded({ blob: hostedBlob, object: holder, metrics });
        setHostedError(null);
      } catch {
        if (!cancelled) { setLoaded(null); setHostedError("This hosted model is too detailed (or not readable) to preview here. You can still download the GLB."); }
      }
    })();
    return () => { cancelled = true; };
  }, [hostedBlob]);
  const hostedReady = hostedBlob && loaded?.blob === hostedBlob ? loaded : null;
  const showHosted = Boolean(hostedBlob);
  const summary = useMemo(() => (spec ? describeSpec(spec) : { levels: 0, volumes: 0, footprint: "" }), [spec]);
  const layout = useMemo(() => (spec ? computeLayout(spec) : { slabs: [], bounds: { min: [0, 0, 0] as [number, number, number], max: [1, 1, 1] as [number, number, number] }, floors: 0, warnings: [] as string[] }), [spec]);
  const complexity = useMemo(() => layoutComplexity(layout), [layout]);
  const metrics = useMemo(() => (showHosted ? hostedReady?.metrics ?? metricsFromSize([1, 1, 1]) : sceneMetrics(layout)), [showHosted, hostedReady, layout]);

  const closeFocus = useCallback(() => { setFocus(false); focusButton.current?.focus(); }, []);
  useEffect(() => {
    if (!focus) return;
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && closeFocus();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [focus, closeFocus]);
  // Bounds-based framing follows the panel when it grows or shrinks.
  useEffect(() => { const id = window.setTimeout(() => setFrameToken((value) => value + 1), 80); return () => window.clearTimeout(id); }, [focus]);

  const exportPng = () => {
    const canvas = canvasWrap.current?.querySelector("canvas");
    setPngError(null);
    if (!canvas) return setPngError("There is no 3D view to capture right now.");
    canvas.toBlob((blob) => (blob ? download(blob, "sift-study.png") : setPngError("The browser could not capture the view as a PNG.")), "image/png");
  };

  const exportGlb = async () => {
    if (hosted) { download(hosted.blob, "sift-hosted-model.glb"); setExportState("complete"); return; }
    if (!spec) return;
    if (!complexity.ok) { setExportState("error"); return; }
    setExportState("exporting");
    const group: Group = buildBuildingGroup(spec); // always the shaded model, independent of the viewer mode
    try {
      const { GLTFExporter } = await import("three/examples/jsm/exporters/GLTFExporter.js");
      const result = await new GLTFExporter().parseAsync(group, { binary: true });
      if (!(result instanceof ArrayBuffer)) throw new Error("Expected a binary GLB export.");
      download(new Blob([result], { type: "model/gltf-binary" }), "sift-massing.glb");
      setExportState("complete");
    } catch {
      setExportState("error");
    } finally {
      disposeBuildingGroup(group);
    }
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    const step = event.shiftKey ? 0.2 : 0.08;
    const handled = { ArrowLeft: () => rig.current?.orbit(-step, 0), ArrowRight: () => rig.current?.orbit(step, 0), ArrowUp: () => rig.current?.orbit(0, -step), ArrowDown: () => rig.current?.orbit(0, step), "+": () => rig.current?.zoom(1.15), "=": () => rig.current?.zoom(1.15), "-": () => rig.current?.zoom(1 / 1.15), f: () => setFrameToken((v) => v + 1), F: () => setFrameToken((v) => v + 1) }[event.key];
    if (handled) { event.preventDefault(); event.stopPropagation(); handled(); }
  };

  const resetView = () => { setPreset("perspective"); setFrameToken((value) => value + 1); };

  return (
    <section className={`preview-panel ${focus ? "preview-panel--focus" : ""}`} aria-label="3D study preview" role={focus ? "dialog" : "complementary"} aria-modal={focus ? true : undefined}>
      <header className="preview-panel__header">
        <div>
          <span className="section-kicker">Live study</span>
          <h2>Form preview</h2>
        </div>
        <span className="provider-chip"><i /> {provider === "procedural" ? "Local fallback" : `${providerLabel(provider)} preview`}</span>
        <button ref={focusButton} type="button" className="viewer-focus" onClick={() => (focus ? closeFocus() : setFocus(true))} aria-pressed={focus}>{focus ? "Close focus (Esc)" : "Focus"}</button>
      </header>

      <div className="viewer-bar" role="toolbar" aria-label="Camera and display">
        <div className="viewer-bar__group" role="group" aria-label="Camera preset">
          {CAMERA_PRESETS.map((item) => <button type="button" key={item} aria-pressed={preset === item} onClick={() => setPreset(item)}>{PRESET_LABELS[item]}</button>)}
        </div>
        <div className="viewer-bar__group" role="group" aria-label="Display mode">
          {VIEW_MODES.map((item) => <button type="button" key={item} aria-pressed={mode === item} disabled={showHosted && item !== "shaded"} title={showHosted && item !== "shaded" ? "Display modes apply to procedural models only" : undefined} onClick={() => setMode(item)}>{MODE_LABELS[item]}</button>)}
        </div>
        <div className="viewer-bar__group" role="group" aria-label="Scene helpers">
          <Toggle label="Grid" pressed={grid} onChange={setGrid} />
          <Toggle label="Axes" pressed={axes} onChange={setAxes} />
          <Toggle label="Shadows" pressed={shadows} onChange={setShadows} />
        </div>
      </div>

      <div className="preview-panel__canvas" ref={canvasWrap} tabIndex={0} role="application" aria-label="3D viewport. Arrow keys orbit, plus and minus zoom, F frames the model." onKeyDown={onKeyDown} onWheel={(event) => event.stopPropagation()} onPointerDown={(event) => event.stopPropagation()}>
        {!webglSupported ? (
          <ViewerFallback title="3D preview unavailable" message="This browser or device cannot create a WebGL context, so the 3D view and PNG renders are disabled. Your project, inspector edits, and GLB download still work." actions={spec && complexity.ok ? [{ label: "Download GLB", onClick: () => void exportGlb() }] : []} />
        ) : !complexity.ok && !showHosted ? (
          <ViewerFallback title="Model too complex to preview" message={complexity.message ?? "This building exceeds the preview limits."} />
        ) : (
          <>
            <ViewerBoundary resetKey={canvasKey} onReset={() => setCanvasKey((value) => value + 1)}>
        <Canvas key={canvasKey} frameloop="demand" dpr={[1, 1.75]} shadows={shadows ? "basic" : false} gl={{ antialias: true, preserveDrawingBuffer: true, powerPreference: "high-performance" }} camera={{ position: [56, 42, 56], fov: PERSPECTIVE_FOV }} onCreated={({ gl, invalidate }) => { exposeDiagnostics(gl); const canvas = gl.domElement; canvas.addEventListener("webglcontextlost", (event) => { event.preventDefault(); setContextLost(true); }); canvas.addEventListener("webglcontextrestored", () => { setContextLost(false); invalidate(); }); }}>
          <color attach="background" args={["#d9d3c6"]} />
          <fog attach="fog" args={["#d9d3c6", 70, 160]} />
          <ambientLight intensity={mode === "wireframe" ? 0.4 : 1.1} />
          <directionalLight position={[18, 34, 20]} intensity={2.2} castShadow={shadows} shadow-mapSize={[2048, 2048]} shadow-camera-left={-34} shadow-camera-right={34} shadow-camera-top={34} shadow-camera-bottom={-34} shadow-camera-near={1} shadow-camera-far={120} shadow-bias={-0.0004} />
          {showHosted ? hostedReady && <primitive object={hostedReady.object} /> : spec && <Model spec={spec} mode={mode} />}
          {grid && <gridHelper args={[100, 40, "#aaa396", "#c7c1b4"]} position={[0, -0.51, 0]} />}
          {axes && <axesHelper args={[Math.max(10, metrics.size[0])]} position={[-metrics.size[0] / 2 - 2, 0, metrics.size[2] / 2 + 2]} />}
          <CameraRig preset={preset} metrics={metrics} frameToken={frameToken} onRig={onRig} />
        </Canvas>
            </ViewerBoundary>
            {contextLost && <ViewerFallback title="Graphics context lost" message="The browser reclaimed the graphics card (often after many tabs or a sleep). Your project is safe." actions={[{ label: "Restore 3D view", onClick: () => { setContextLost(false); setCanvasKey((value) => value + 1); } }]} />}
          </>
        )}
        <div className="preview-panel__caption">
          {showHosted ? <><span>{hosted?.label}</span><span>not editable</span><span>unverified</span></> : <><span>{summary.levels} levels</span><span>{summary.volumes} {summary.volumes === 1 ? "volume" : "volumes"}</span><span>{summary.footprint}</span></>}
        </div>
        <p className="viewer-hint" aria-hidden="true">Drag orbit · right-drag pan · wheel zoom · arrows/+/−/F by keyboard</p>
      </div>

      <div className="preview-panel__actions">
        <button type="button" onClick={() => setFrameToken((value) => value + 1)}>Frame model</button>
        <button type="button" onClick={resetView}>Reset view</button>
        <button type="button" onClick={exportPng}>PNG</button>
        <button type="button" onClick={() => void exportGlb()} disabled={exportState === "exporting"}>
          {exportState === "exporting" ? "Exporting…" : exportState === "complete" ? "GLB saved" : exportState === "error" ? "Retry GLB" : "GLB"}
        </button>
      </div>
      {pngError && <p className="preview-panel__note" role="alert">{pngError}</p>}
      {exportState === "error" && !complexity.ok && <p className="preview-panel__note" role="alert">GLB export was blocked because the model exceeds the complexity limit.</p>}
      {hostedError && <p className="preview-panel__note" role="alert">{hostedError}</p>}
      {showHosted && <p className="preview-panel__note">Hosted mesh ({hosted?.label.replace(/ GLB$/, "")}) — a fixed model, not editable geometry. Hosted generation is unverified against a live account.</p>}
      {stale && <p className="preview-panel__note" role="status">Showing the last generated model — the prompt has changed since. Run the Generation node again.</p>}
      {layout.warnings.length > 0 && <p className="preview-panel__note" role="status">{layout.warnings[0]}</p>}
      <p className="preview-panel__note">Concept massing only — not BIM, engineering, or construction geometry.</p>
    </section>
  );
}

"use client";

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Group } from "three";
import type { OrbitControls as OrbitControlsType } from "three/examples/jsm/controls/OrbitControls.js";
import type { MassingSpec, Provider } from "@/lib/contracts";
import { buildMassingGroup, disposeMassingGroup } from "@/lib/three-massing";

function CameraControls({ resetToken }: { resetToken: number }) {
  const { camera, gl } = useThree();
  const controls = useRef<OrbitControlsType | null>(null);

  useEffect(() => {
    let mounted = true;
    void import("three/examples/jsm/controls/OrbitControls.js").then(({ OrbitControls }) => {
      if (!mounted) return;
      const instance = new OrbitControls(camera, gl.domElement);
      instance.enableDamping = true;
      instance.dampingFactor = 0.07;
      instance.maxPolarAngle = Math.PI * 0.49;
      instance.target.set(0, 9, 0);
      instance.update();
      controls.current = instance;
    });
    return () => {
      mounted = false;
      controls.current?.dispose();
    };
  }, [camera, gl]);

  useEffect(() => {
    camera.position.set(56, 42, 56);
    controls.current?.target.set(0, 9, 0);
    controls.current?.update();
  }, [camera, resetToken]);

  useFrame(() => controls.current?.update());
  return null;
}

function MassingModel({ spec }: { spec: MassingSpec }) {
  const group = useMemo(() => buildMassingGroup(spec), [spec]);
  useEffect(() => () => disposeMassingGroup(group), [group]);
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

export function ModelPreview({ spec, provider }: { spec: MassingSpec; provider: Provider }) {
  const canvasWrap = useRef<HTMLDivElement>(null);
  const [resetToken, setResetToken] = useState(0);
  const [exportState, setExportState] = useState<"idle" | "exporting" | "complete" | "error">("idle");

  const exportPng = () => {
    const canvas = canvasWrap.current?.querySelector("canvas");
    canvas?.toBlob((blob) => blob && download(blob, "sift-study.png"), "image/png");
  };

  const exportGlb = async () => {
    setExportState("exporting");
    const group: Group = buildMassingGroup(spec);
    try {
      const { GLTFExporter } = await import("three/examples/jsm/exporters/GLTFExporter.js");
      const result = await new GLTFExporter().parseAsync(group, { binary: true });
      if (!(result instanceof ArrayBuffer)) throw new Error("Expected a binary GLB export.");
      download(new Blob([result], { type: "model/gltf-binary" }), "sift-massing.glb");
      setExportState("complete");
    } catch {
      setExportState("error");
    } finally {
      disposeMassingGroup(group);
    }
  };

  return (
    <aside className="preview-panel" aria-label="3D study preview">
      <header className="preview-panel__header">
        <div>
          <span className="section-kicker">Live study</span>
          <h2>Form preview</h2>
        </div>
        <span className="provider-chip"><i /> {provider === "procedural" ? "Local fallback" : "Meshy preview"}</span>
      </header>
      <div className="preview-panel__canvas" ref={canvasWrap}>
        <Canvas camera={{ position: [56, 42, 56], fov: 38 }} shadows="basic" gl={{ antialias: true, preserveDrawingBuffer: true }}>
          <color attach="background" args={["#d9d3c6"]} />
          <fog attach="fog" args={["#d9d3c6", 55, 110]} />
          <ambientLight intensity={1.1} />
          <directionalLight position={[18, 34, 20]} intensity={2.2} castShadow />
          <MassingModel spec={spec} />
          <gridHelper args={[100, 40, "#aaa396", "#c7c1b4"]} position={[0, -0.51, 0]} />
          <CameraControls resetToken={resetToken} />
        </Canvas>
        <div className="preview-panel__caption">
          <span>{spec.floors} levels</span><span>{Math.round(spec.width)} × {Math.round(spec.depth)} m</span><span>{spec.material}</span>
        </div>
      </div>
      <div className="preview-panel__actions">
        <button type="button" onClick={() => setResetToken((value) => value + 1)}>Reset view</button>
        <button type="button" onClick={exportPng}>PNG</button>
        <button type="button" onClick={() => void exportGlb()} disabled={exportState === "exporting"}>
          {exportState === "exporting" ? "Exporting…" : exportState === "complete" ? "GLB saved" : exportState === "error" ? "Retry GLB" : "GLB"}
        </button>
      </div>
      <p className="preview-panel__note">Concept massing only — not BIM, engineering, or construction geometry.</p>
    </aside>
  );
}


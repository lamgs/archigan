import { Box3 } from "three";
import { describe, expect, it } from "vitest";
import { computeLayout } from "./geometry";
import { buildBuildingGroup, disposeBuildingGroup } from "./three-building";
import { deriveBuildingSpec } from "./typologies";
import { CAMERA_PRESETS, PERSPECTIVE_FOV, TARGET_SIZE, VIEW_MODES, framingDistance, presetPose, sceneMetrics } from "./viewer";

const briefs = ["A terraced stepped office tower with a public podium", "Twin towers rising from a shared podium", "A cylindrical glass residential tower", "A low concrete archive pavilion"];
const viewport = { width: 800, height: 600 };

describe("sceneMetrics", () => {
  it.each(briefs)("matches the geometry the builder produces: %s", (brief) => {
    const spec = deriveBuildingSpec(brief);
    const layout = computeLayout(spec);
    const metrics = sceneMetrics(layout);
    const group = buildBuildingGroup(spec, layout);
    // Ignore the ground plate (the last child) and measure the building only.
    const building = group.children.slice(0, -1);
    const box = new Box3();
    building.forEach((child) => box.expandByObject(child));
    expect(Math.max(...metrics.size)).toBeCloseTo(TARGET_SIZE, 5);
    expect(box.max.y - box.min.y).toBeCloseTo(metrics.size[1], 1);
    expect((box.max.y + box.min.y) / 2).toBeCloseTo(metrics.center[1], 1);
    expect(box.max.x - box.min.x).toBeLessThanOrEqual(metrics.size[0] + 0.3);
    disposeBuildingGroup(group);
  });
});

describe("framing", () => {
  it("places the camera far enough that the bounding sphere fits both axes", () => {
    [0.5, 1, 1.6, 2.4].forEach((aspect) => {
      const radius = 20;
      const distance = framingDistance(radius, PERSPECTIVE_FOV, aspect);
      const vertical = (PERSPECTIVE_FOV * Math.PI) / 180;
      const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * aspect);
      expect(distance * Math.sin(vertical / 2)).toBeGreaterThanOrEqual(radius);
      expect(distance * Math.sin(horizontal / 2)).toBeGreaterThanOrEqual(radius);
    });
  });
  it("frames taller models from further away", () => {
    const small = sceneMetrics(computeLayout(deriveBuildingSpec("A low concrete archive pavilion")));
    const tall = sceneMetrics(computeLayout(deriveBuildingSpec("A slender glass tower with a spiral twist")));
    expect(tall.radius).toBeGreaterThan(0);
    expect(presetPose("perspective", tall, viewport).position[1]).not.toBe(presetPose("perspective", small, viewport).position[1]);
  });
});

describe("camera presets", () => {
  const metrics = sceneMetrics(computeLayout(deriveBuildingSpec(briefs[1])));
  const dir = (preset: (typeof CAMERA_PRESETS)[number]) => {
    const pose = presetPose(preset, metrics, viewport);
    const d = pose.position.map((v, i) => v - pose.target[i]);
    const len = Math.hypot(...d);
    return { pose, unit: d.map((v) => v / len) };
  };
  it("offers all five presets with the right projection and look direction", () => {
    expect(CAMERA_PRESETS).toEqual(["perspective", "axonometric", "top", "front", "right"]);
    expect(dir("top").unit).toEqual([0, 1, 0]);
    expect(dir("front").unit).toEqual([0, 0, 1]);
    expect(dir("right").unit).toEqual([1, 0, 0]);
    expect(dir("top").pose.projection).toBe("orthographic");
    expect(dir("perspective").pose.projection).toBe("perspective");
    const axo = dir("axonometric");
    expect(axo.pose.projection).toBe("orthographic");
    expect(axo.unit[0]).toBeCloseTo(axo.unit[2]);
    expect(axo.unit[1]).toBeGreaterThan(0);
  });
  it("always targets the model centre and gives orthographic views a usable zoom", () => {
    CAMERA_PRESETS.forEach((preset) => {
      const pose = presetPose(preset, metrics, viewport);
      expect(pose.target).toEqual(metrics.center);
      if (pose.projection === "orthographic") expect(pose.zoom).toBeGreaterThan(1);
    });
  });
  it("fits the plan in the top view within the viewport", () => {
    const pose = presetPose("top", metrics, viewport);
    expect(metrics.size[0] * (pose.zoom ?? 0)).toBeLessThanOrEqual(viewport.width);
    expect(metrics.size[2] * (pose.zoom ?? 0)).toBeLessThanOrEqual(viewport.height);
  });
  it("only allows free orbit for perspective and axonometric views", () => {
    expect(CAMERA_PRESETS.filter((p) => presetPose(p, metrics, viewport).orbit)).toEqual(["perspective", "axonometric"]);
  });
});

describe("view modes", () => {
  it.each(VIEW_MODES)("builds a valid group in %s mode", (mode) => {
    const spec = deriveBuildingSpec(briefs[2]);
    const group = buildBuildingGroup(spec, undefined, mode);
    const materials = new Set<unknown>();
    group.traverse((child) => { const m = (child as { material?: unknown }).material; if (m) materials.add(m); });
    expect(group.children.length).toBeGreaterThan(10);
    if (mode === "wireframe") [...materials].slice(0, -1).forEach((m) => expect((m as { wireframe?: boolean }).wireframe).toBe(true));
    if (mode === "clay") expect(new Set([...materials].map((m) => (m as { color: { getHexString: () => string } }).color.getHexString())).size).toBeLessThanOrEqual(3);
    disposeBuildingGroup(group);
  });
});

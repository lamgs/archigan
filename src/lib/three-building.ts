import * as THREE from "three";
import type { BuildingSpec } from "./contracts";
import { computeLayout, type Layout, type Slab } from "./geometry";

const TARGET_SIZE = 28;

function makeMaterial(spec: BuildingSpec, id: string) {
  const def = spec.materials[id];
  const glass = def?.kind === "glass";
  return new THREE.MeshStandardMaterial({
    color: def?.color ?? "#999999",
    roughness: glass ? 0.18 : def?.kind === "metal" ? 0.45 : 0.82,
    metalness: glass ? 0.08 : def?.kind === "metal" ? 0.6 : 0.02,
    transparent: glass,
    opacity: glass ? 0.72 : 1,
  });
}

function geometryFor(slab: Slab, height: number, grow = 0) {
  if (slab.shape.type === "circle") return new THREE.CylinderGeometry(slab.shape.radius + grow, slab.shape.radius + grow, height, 40);
  return new THREE.BoxGeometry(slab.shape.width + grow * 2, height, slab.shape.depth + grow * 2);
}

export function buildBuildingGroup(spec: BuildingSpec, layout: Layout = computeLayout(spec)) {
  const group = new THREE.Group();
  group.name = "Sift architectural building";
  const size = layout.bounds.max.map((value, axis) => value - layout.bounds.min[axis]);
  const scale = TARGET_SIZE / Math.max(1, ...size);
  const materials = new Map<string, THREE.MeshStandardMaterial>();
  const material = (id: string) => materials.get(id) ?? (materials.set(id, makeMaterial(spec, id)), materials.get(id) as THREE.MeshStandardMaterial);
  const glazingMaterial = new THREE.MeshStandardMaterial({ color: "#86a9aa", roughness: 0.12, metalness: 0.1, transparent: true, opacity: 0.7 });

  const centreX = (layout.bounds.min[0] + layout.bounds.max[0]) / 2;
  const centreZ = (layout.bounds.min[2] + layout.bounds.max[2]) / 2;

  layout.slabs.forEach((slab) => {
    const place = (mesh: THREE.Mesh, y: number) => {
      mesh.position.set((slab.x - centreX) * scale, y * scale, (slab.z - centreZ) * scale);
      mesh.rotation.y = slab.rotationY;
      mesh.scale.setScalar(scale);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
    };
    const band = slab.kind === "floor" ? slab.height * slab.glazing : 0;
    const solid = slab.height - band;
    if (solid > 0) place(new THREE.Mesh(geometryFor(slab, solid), material(slab.materialId)), slab.y + solid / 2);
    if (band > 0) place(new THREE.Mesh(geometryFor(slab, band, 0.04), glazingMaterial), slab.y + solid + band / 2);
  });

  const ground = new THREE.Mesh(new THREE.BoxGeometry(46, 0.5, 46), new THREE.MeshStandardMaterial({ color: "#5d5a52", roughness: 0.96 }));
  ground.position.y = -0.28;
  ground.receiveShadow = true;
  group.add(ground);
  return group;
}

export function disposeBuildingGroup(group: THREE.Group) {
  const materials = new Set<THREE.Material>();
  group.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    (Array.isArray(child.material) ? child.material : [child.material]).forEach((m) => materials.add(m));
  });
  materials.forEach((m) => m.dispose());
}

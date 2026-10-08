import * as THREE from "three";
import type { MassingSpec } from "./contracts";

const palette: Record<MassingSpec["material"], { color: string; roughness: number; metalness: number; opacity?: number }> = {
  limestone: { color: "#d8d0bf", roughness: 0.78, metalness: 0.02 },
  terracotta: { color: "#a45139", roughness: 0.72, metalness: 0.01 },
  concrete: { color: "#858681", roughness: 0.9, metalness: 0 },
  glass: { color: "#86a9aa", roughness: 0.18, metalness: 0.08, opacity: 0.72 },
};

function addVolume(group: THREE.Group, size: [number, number, number], position: [number, number, number], material: THREE.Material) {
  const geometry = new THREE.BoxGeometry(...size);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  group.add(mesh);
}

export function buildMassingGroup(spec: MassingSpec) {
  const group = new THREE.Group();
  group.name = "Sift architectural massing";
  const finish = palette[spec.material];
  const material = new THREE.MeshStandardMaterial({
    color: finish.color,
    roughness: finish.roughness,
    metalness: finish.metalness,
    transparent: Boolean(finish.opacity),
    opacity: finish.opacity ?? 1,
  });

  const maxDimension = Math.max(spec.width, spec.depth, spec.floors * spec.floorHeight);
  const scale = 28 / maxDimension;

  for (let floor = 0; floor < spec.floors; floor += 1) {
    const progress = spec.floors === 1 ? 0 : floor / (spec.floors - 1);
    const width = Math.max(spec.width * (1 - spec.terrace * floor), spec.width * 0.52) * scale;
    const depth = Math.max(spec.depth * (1 - spec.terrace * floor * 0.72), spec.depth * 0.58) * scale;
    const height = spec.floorHeight * scale * 0.91;
    const y = floor * spec.floorHeight * scale + height / 2;
    const x = Math.sin(progress * Math.PI * 2) * spec.twist * floor * scale * 0.7;
    const z = Math.cos(progress * Math.PI * 2) * spec.twist * floor * scale * 0.7;
    const rotation = spec.twist * progress;

    const floorGroup = new THREE.Group();
    floorGroup.position.set(x, y, z);
    floorGroup.rotation.y = rotation;

    if (spec.courtyard && width > 9 && depth > 8) {
      const bar = Math.min(width, depth) * 0.2;
      addVolume(floorGroup, [width, height, bar], [0, 0, (depth - bar) / 2], material);
      addVolume(floorGroup, [width, height, bar], [0, 0, -(depth - bar) / 2], material);
      addVolume(floorGroup, [bar, height, depth - bar * 2], [(width - bar) / 2, 0, 0], material);
      addVolume(floorGroup, [bar, height, depth - bar * 2], [-(width - bar) / 2, 0, 0], material);
    } else {
      addVolume(floorGroup, [width, height, depth], [0, 0, 0], material);
    }
    group.add(floorGroup);
  }

  const baseMaterial = new THREE.MeshStandardMaterial({ color: "#5d5a52", roughness: 0.96 });
  addVolume(group, [34, 0.5, 34], [0, -0.28, 0], baseMaterial);
  return group;
}

export function disposeMassingGroup(group: THREE.Group) {
  group.traverse((child) => {
    if (!(child instanceof THREE.Mesh)) return;
    child.geometry.dispose();
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    materials.forEach((material) => material.dispose());
  });
}


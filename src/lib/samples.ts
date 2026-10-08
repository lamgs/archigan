import type { SiftProjectV2 } from "./contracts";
import { createWorkflowProject } from "./projects";

const NOW = "2026-10-08T12:00:00.000Z";
const sample = (id: string, name: string, prompt: string, refinement: string): SiftProjectV2 => createWorkflowProject({ id, name, now: NOW, prompt, refinement });

export const sampleProjects = [
  sample("sample-courtyard", "Courtyard Commons", "A warm terracotta arts center organized around a shaded courtyard", "Step the upper floors into planted terraces"),
  sample("sample-tower", "Spiral Habitat", "A slender glass residential tower with a gentle twist", "Create a generous public base and a rooftop garden"),
  sample("sample-pavilion", "River Archive", "A low concrete archive pavilion stretched along the river", "Carve an atrium void and lift the entry canopy"),
];

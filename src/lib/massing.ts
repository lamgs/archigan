export function normalizeBrief(prompt: string, refinement = "") {
  return `${prompt.trim().replace(/\s+/g, " ")} ${refinement.trim().replace(/\s+/g, " ")}`
    .trim()
    .toLowerCase();
}

export function composeArchitecturalPrompt(prompt: string, refinement = "") {
  const clean = normalizeBrief(prompt, refinement);
  return `Architectural concept massing model of ${clean}. Standalone building, clean geometry, coherent structure, no people, no vehicles, no text, neutral presentation.`;
}


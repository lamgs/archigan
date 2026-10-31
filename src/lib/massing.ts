export function normalizeBrief(prompt: string, refinement = "") {
  return `${prompt.trim().replace(/\s+/g, " ")} ${refinement.trim().replace(/\s+/g, " ")}`
    .trim()
    .toLowerCase();
}

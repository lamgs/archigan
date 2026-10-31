/**
 * Architectural steering for hosted text-to-3D. Generic text-to-3D models are trained on the whole world's objects, so a
 * bare "wooden cabin" drifts toward vernacular / storybook buildings. A text model cannot be pointed at a database of
 * awarded projects; the only levers are (1) positive vocabulary that contemporary architecture publications use and
 * (2) a negative prompt that suppresses the vernacular, ornamental and toy-like look. Both are plain constants so they are
 * easy to tune after looking at real output. Whether this actually improves Tripo's results is UNVERIFIED (no live runs).
 */

/** Priority-ordered clauses: the brief always comes first, then as many clauses as fit the provider's prompt limit. */
export const STYLE_CLAUSES = [
  "contemporary architect-designed building in the manner of recent award-winning projects published on ArchDaily and Dezeen",
  "restrained minimalist composition with strong simple massing and considered proportions",
  "crisp orthogonal geometry, flat or very low-pitched roof planes, deep overhangs, cantilevers",
  "large frameless glazing and clean planar facades",
  "refined material palette such as board-formed concrete, vertical or charred timber cladding, weathering steel and glass",
  "a single isolated building on a plain neutral base",
];

/** Terms to suppress. A term is dropped if the user's own brief uses one of its words, so we never fight explicit intent. */
export const NEGATIVE_TERMS = [
  "vernacular", "traditional", "historic", "gothic", "medieval", "rustic", "village house", "cottage", "chalet",
  "ornate", "decorative ornament", "steep gabled roof", "thatched roof", "fantasy", "cartoon", "toy", "low-poly game asset",
  "clutter", "trees", "people", "vehicles", "text", "interior",
];

const words = (value: string) => value.toLowerCase().match(/[a-z]{4,}/g) ?? [];

/** Negative prompt for a brief, minus any term the brief itself asks for (e.g. a brief saying "rustic" keeps "rustic"). */
export function negativePromptFor(brief: string): string {
  const used = new Set(words(brief));
  return NEGATIVE_TERMS.filter((term) => !words(term).some((w) => used.has(w))).join(", ");
}

/** Brief first (case and wording preserved), then as many style clauses as fit `max` characters. */
export function composeArchitecturalPrompt(prompt: string, refinement = "", max = 1024): string {
  const brief = [prompt, refinement].map((part) => part.trim().replace(/\s+/g, " ")).filter(Boolean).join(" ").replace(/[.\s]+$/, "");
  let text = brief;
  for (const clause of STYLE_CLAUSES) {
    const next = `${text}${text === brief ? ". " : ", "}${clause}`;
    if (next.length + 1 > max) break;
    text = next;
  }
  return `${text}.`;
}

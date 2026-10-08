# Design references

These references make the intended direction durable for agents that cannot browse. They are observations, not instructions to copy proprietary UI.

## xFigura cues to adapt

Official xFigura documentation describes a single canvas connecting nodes across creative workflows. The useful cues for Sift are spatial workflow legibility, dark neutral surfaces, compact tool chrome, visible node lineage, and previewing output without leaving the canvas.

Sift narrows that pattern:

- a four-stage left-to-right graph: brief → massing → refine → export;
- a persistent architectural 3D preview beside the graph;
- warm ivory canvas, ink typography, oxblood accents, and drawing-board grid rather than a generic dark AI dashboard;
- explicit provider/fallback state and architectural caveats;
- no video, image-generation maze, or unrelated model catalog.

## Saved visual map

```text
┌ project rail ┐ ┌──────────── workflow canvas ────────────┐ ┌── 3D study ──┐
│ SIFT         │ │ [BRIEF] → [MASSING] → [REFINE] → [OUT] │ │ live massing │
│ samples      │ │                                          │ │ orbit/export │
│ saved work   │ │       generous drawing-board space       │ │ provider tag │
└──────────────┘ └──────────────────────────────────────────┘ └──────────────┘
                       ┌── prompt / refinement dock ──┐
                       └───────────────────────────────┘
```

## Source snapshots

- xFigura documentation, checked 2026-10-08: <https://xfigura.gitbook.io/xfigura-docs>
- Meshy Text-to-3D API, checked 2026-10-08: <https://docs.meshy.ai/en/api/text-to-3d>

If locally captured screenshots are added later, place them under `docs/reference/`, record source and capture date, and use them only for design study.


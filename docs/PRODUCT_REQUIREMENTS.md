# Sift 2.0 product requirements

This document is the durable, implementation-oriented version of the Sift 2.0 master brief supplied on 2026-10-08. It defines product scope and acceptance. `STATUS.md` records what is true now; `TASKS.md` records the executable backlog; `IMPLEMENTATION_PLAN.md` defines sequence and milestone gates.

## Product statement

Sift is an AI-assisted architectural design workspace that transforms natural-language design intent into explorable 3D forms, allowing designers to generate, refine, visualize, and compare building concepts through a connected canvas.

Sift is an independent project inspired by modern AI design workflows. It is not an official xFigura product or integration, and it does not claim to run the original ArchiGAN model.

## Core outcome

A user can move through this complete workflow:

```text
Prompt → 3D generation → interactive model → refine / branch → render → export
```

The output must contain real geometry. Static images, fake progress, unrelated hard-coded models, empty exports, and simulated provider success are not acceptable.

## MVP requirements

### Projects and canvas

- Project dashboard with create, open, rename, delete, and sample-project actions.
- Infinite React Flow canvas with movable nodes, meaningful typed connections, zoom, pan, fit-to-content, minimap, and contextual node creation.
- Prompt, generation, model, render, and variation node types.
- Graph lineage survives save/reopen; unsupported connections and accidental execution cycles are rejected.
- Compact left toolbar and collapsible contextual properties inspector.

### Procedural architecture

- Credential-free generation of real architectural geometry from a validated `BuildingSpec`.
- Rectangular and cylindrical footprints; single and multiple volumes; podium/tower compositions; floors and floor height; setbacks/terraces; taper, rotation, and offsets; simple facade patterns; roof treatment; material assignment.
- A constrained local prompt parser may be used, but the UI must not imply arbitrary language understanding.
- Direct parameter editing visibly changes geometry while preserving the prior version.
- At least three materially distinct typologies: terraced podium/tower, twin towers/shared podium, and cylindrical or rotated tower.

### 3D viewing and rendering

- R3F/Three.js viewer with real geometry, studio lighting, shadows, neutral background, ground plane, antialiasing, appropriate tone mapping, and automatic bounds-based framing.
- Orbit, pan, zoom, reset/frame, and perspective/top/front/right/axonometric camera presets.
- Grid, axes, ground-shadow toggles; clay, architectural shaded, glass/concrete, and wireframe material modes.
- Expanded/focus viewer that prevents pointer conflicts with the outer React Flow canvas.
- Render node/output using the actual selected model, camera, material, lighting, background, and resolution.
- PNG support for 1024×1024 and 1600×900, plus 1920×1080 when device limits allow.
- Valid GLB export that can be reopened in a compatible viewer and matches the visible model.

### Iteration and branching

- Follow-up prompts and parameter edits create child variants; originals are never silently overwritten.
- At least two revisions can branch from one source and remain simultaneously visible.
- Connections communicate real lineage and restore correctly after reopening.
- Procedural variants modify validated specifications. Hosted meshes are regenerated or edited only through provider-supported operations, with the behavior labeled accurately.

### Persistence

- IndexedDB stores project state and model/render assets where appropriate; large blobs never go into localStorage.
- Restore board viewport, nodes, positions, edges, prompts, specifications, settings, lineage, asset references, and known provider jobs.
- Loading, progress, empty, failure, timeout, rate-limit, missing-credential, unsupported-WebGL, and recovery states are meaningful and honest.

### Hosted generation

- Optional server-side hosted adapter (Tripo, ADR-018); credentials never enter the client bundle or local project data.
- Explicit paid-generation confirmation naming the provider before a request.
- Validated and protected create request; asynchronous job ID; queued/running/completed/failed/timed-out/rate-limited/cancelled states; polling or supported streaming; reload-safe job state.
- Download completed GLB assets into persistent storage rather than relying indefinitely on expiring signed URLs.
- Without credentials, hosted mode is disabled with setup guidance and procedural mode remains functional.
- Live behavior is marked verified only after a real-account test. Documented-contract and mock tests must be labeled as such.

### Portfolio quality

- Product name `Sift` and descriptor `AI Architectural Form Studio`.
- Cohesive, canvas-first architectural visual identity; minimal chrome; precise typography and spacing; no copied xFigura branding or proprietary assets.
- Refined first-run experience, honest capability copy, high-quality samples, responsive layout, accessibility, and usable error recovery.
- README covers architecture, generation model, setup, optional credentials, testing, deployment, capabilities, and limitations.

## Bundled example

The principal sample is **Terraced Tower Study**:

> Create a 12-story mixed-use building consisting of a four-story rectangular podium and an eight-story tower above. Introduce stepped setbacks every two floors, generous terraces, and a contemporary glazed facade.

It must work without credentials and demonstrate a structured specification, recognizable podium/tower geometry, floor differentiation, setbacks, materials, expanded viewer, two retained variations, PNG render, and valid GLB. Additional samples cover twin towers/shared podium and a cylindrical or rotated residential tower.

## Target architecture

The system is divided into UI, domain, geometry, provider, and persistence layers. React components do not consume vendor responses or IndexedDB directly.

Canonical domain objects:

- `BuildingSpec`: validated architectural parameters and volumes.
- `DesignNode`: typed prompt/generation/model/render/variation graph node with typed ports.
- `Artifact`: immutable building-spec, model-GLB, or render-PNG output.
- `GenerationJob`: provider-neutral async execution state separated from artifact state.
- `DesignRevision`: parent/child lineage and the instruction or parameter change that produced it.
- `SiftProject`: versioned aggregate containing graph, board viewport, settings, references, and migrations.

Detailed target contracts and migration rules live in `ARCHITECTURE.md`.

## Explicit exclusions

- Video generation.
- General text-to-image, image editing, or upscaling platforms.
- Model training, dataset collection, custom GAN/diffusion pipelines, or proprietary-data dependency.
- BIM documentation, full CAD/NURBS, structural engineering, code compliance, or fabrication guarantees.
- Rhino, Revit, or Grasshopper plugins.
- Collaboration/multiplayer, payments/subscriptions, or a broad model marketplace.

## Acceptance suite

The MVP is done only when the following are automated where practical and manually evidenced where browser/GPU behavior requires it:

1. Procedural prompt creates a new generation/model artifact with real parameter-corresponding geometry.
2. Orbit, pan, zoom, reset, and camera presets work without moving the outer canvas.
3. Floor/dimension changes visibly alter geometry and retain the previous version.
4. Two child revisions remain visible with lineage after reload.
5. Selected camera/render mode creates a correct downloadable PNG.
6. GLB contains real matching geometry and reopens successfully.
7. Reload restores the complete board, settings, assets, and jobs.
8. Missing credentials leave procedural mode useful and never fake hosted success.
9. Mock or live hosted jobs handle task IDs, transitions, errors, assets, and reload accurately; live status is reported separately.
10. TypeScript, lint, unit/component tests, Playwright journeys, production build, accessibility checks, and key screenshots pass at the milestone gate.

## Secondary scope after MVP

- Supabase persistence and asset storage.
- A second hosted provider such as Tripo.
- User-uploaded GLB support.
- More advanced materials, environments, presentation rendering, design history, and side-by-side comparison.


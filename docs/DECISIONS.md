# Architecture decision record

## ADR-001 — Preserve rather than repurpose the legacy GAN

**Status:** accepted, 2026-10-08

The existing repository is a 2018-era PyTorch voxel GAN that requires proprietary training data. It remains as historical reference and is not part of Sift's runtime. Sift does not train on or collect new data.

## ADR-002 — Local-first procedural provider

**Status:** accepted, 2026-10-08

The default experience is a deterministic Three.js massing generator. This makes the primary loop instant, free, testable, and useful offline after installation. Hosted AI augments rather than gates the product.

## ADR-003 — Next.js App Router and server-only provider adapters

**Status:** accepted, 2026-10-08

One TypeScript application supplies UI and secure route handlers. Provider keys stay in server environment variables; the browser calls internal routes only.

## ADR-004 — React Flow is the workflow representation

**Status:** accepted, 2026-10-08

The visible graph mirrors the generation lifecycle and leaves room for architectural constraints without becoming a general-purpose multimodal graph editor.

## ADR-005 — IndexedDB before accounts/backends

**Status:** accepted, 2026-10-08

Projects persist locally through an adapter. This minimizes infrastructure and privacy surface during product discovery. The adapter allows later sync without coupling components to IndexedDB.

## ADR-006 — Meshy is optional and currently unverified

**Status:** accepted, 2026-10-08

Official documentation currently exposes asynchronous Text-to-3D preview/refine endpoints and GLB outputs. The adapter contract is documented and server-scoped, but this repository has not exercised it with a real API key. Product copy must say “configured” and “unverified,” never “connected” or “working,” until a live smoke test passes.

## ADR-007 — Focused architectural product boundary

**Status:** accepted, 2026-10-08

Sift borrows xFigura's spatial workflow clarity, not its broad text/image/video/3D scope. Video and unrelated multimodal features are excluded to keep the brief-to-massing loop coherent.

## ADR-008 — Master brief defines the MVP gate

**Status:** accepted, 2026-10-08

The complete Sift 2.0 master brief is distilled in `PRODUCT_REQUIREMENTS.md` and takes precedence over earlier shorthand plans. The existing local vertical slice is a verified foundation, not the completed MVP. MVP status requires validated architectural specifications, direct parameter edits, non-destructive branching, typed artifact lineage, expanded viewer controls, render artifacts, complete restore, provider lifecycle handling, and the ten acceptance scenarios.

## ADR-010 — v2 storage key and legacy-store handling

**Status:** accepted, 2026-10-08

v2 projects persist under `projects-v2`; the legacy `projects-v1` key is read-only and never deleted, so a rollback loses nothing. Records that fail validation are written back untouched instead of dropped. Legacy edges that violate the new port rules are dropped from the migrated graph with a warning (the original stays in `projects-v1`). `BuildingSpec` and the exact legacy `MassingSpec` are held in the building-spec artifact's `metadata` until P0.10 defines geometry storage.

## ADR-009 — Immutable artifacts and non-destructive revisions

**Status:** accepted, 2026-10-08

Building specifications, GLB models, and PNG renders are immutable artifacts. Edits create child revisions connected to their source. Provider job state is mutable execution metadata and is stored separately from artifact state. This separation prevents silent design loss and makes graph lineage and recovery testable.

## ADR-011 — Bundled samples are built from domain functions; featured sample renders itself

**Status:** accepted, 2026-10-09

`samples.ts` builds the featured *Terraced Tower Study* with the same functions the UI uses (`createWorkflowProject`, `branchFrom`, `editNodeGeometry`, `commitVariations`), so samples can never drift from real behaviour and stay deterministic. Samples contain no binary assets: the Render node is flagged `autoRender` and renders once when the sample is opened, so the image always matches the current geometry code. Opening a sample copies it under a new project id (node/artifact ids are only unique per project and are kept). The earlier *Courtyard Commons* sample was removed because courtyard voids are not modelled (it implied a capability that does not exist); *River Archive* and *Spiral Habitat* remain, and *Twin Towers on a Shared Podium* and *Cylindrical Residence* were added so three typologies are demonstrated. Sample wording uses only words the local interpreter honours (see `INTERPRETER_HELP`).

## ADR-012 — Responsive layout stacks below 900 px

**Status:** accepted, 2026-10-09

Below 900 px the board, inspector, and viewer stack vertically (the page scrolls) instead of overlapping; the left rail hides below 1320 px (samples/projects are on the dashboard); the minimap hides on small screens; the inspector collapses to a slim tab until a node is selected. Responsive CSS lives at the end of `globals.css` so it overrides the base rules.


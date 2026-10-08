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


# Sift 2.0 implementation plan

This plan is the shared execution sequence for Codex and Claude Code. Read `PRODUCT_REQUIREMENTS.md` for product scope, `STATUS.md` for the current truth, and `TASKS.md` for checkbox-level acceptance criteria.

Status labels are intentionally strict: **complete** means the milestone exit gate passed; **partial** means useful implementation exists but required acceptance remains.

## Phase 0 — Repository and shared foundation

**Status: complete**

Outcome: agents can resume safely and the legacy research code remains preserved without constraining the modern application.

- Inspect the original PyTorch repository and Git state.
- Establish Next.js/TypeScript structure and versioned shared contracts.
- Add shared agent rules, product requirements, architecture, decisions, tasks, current status, and handoff log.
- Record that the original model/data are unavailable and no new training pipeline is in scope.
- Establish reproducible install, test, lint, audit, and build commands.

Exit gate: a new agent can identify scope, current state, next task, validation commands, and handoff protocol without reconstructing the conversation.

## Phase 1 — Functional 3D foundation

**Status: partial — baseline vertical slice works**

Outcome: a credential-free path produces real architecture, displays it interactively, and exports real artifacts.

Delivered baseline:

- Deterministic prompt-derived `MassingSpec`.
- R3F geometry, lighting, ground/grid context, orbit/pan/zoom/reset.
- Client PNG capture and GLB serialization.
- Three prompt presets and unit coverage for deterministic mapping.

Remaining gate work:

- Replace/extend `MassingSpec` with a versioned `BuildingSpec` supporting rectangular/cylindrical footprints, multiple volumes, podium/tower, offsets, taper, setbacks, facade, roof, and materials.
- Produce three genuinely distinct typologies, not merely parameter variations of stacked boxes.
- Add bounds-based framing, explicit invalid-geometry handling, camera presets, and a reopen-validation test for GLB.
- Add WebGL error boundary/fallback and mesh complexity limits.

Exit gate: all Phase 1 requirements in `TASKS.md` pass for terraced tower, twin towers, and cylindrical/rotated tower.

## Phase 2 — Canvas and product shell

**Status: partial**

Outcome: the complete design process is organized on an understandable infinite canvas.

Delivered baseline:

- React Flow canvas with movable connected lifecycle nodes, dotted grid, controls, minimap, and selection states.
- Responsive shell, sample rail, prompt dock, save status, and persistent 3D preview.

Remaining gate work:

- Project dashboard with new/open/rename/delete/sample flows.
- Add/select/pan toolbar and contextual node menu.
- Prompt, generation, model, render, and variation nodes with typed ports and validated connections.
- Resizing where useful, collapsible properties inspector, and expanded model focus mode.
- Separate the model artifact into a meaningful model node rather than a permanently detached preview panel.
- Persist/restore React Flow viewport as well as graph topology.

Exit gate: project creation through canvas navigation works without hidden developer knowledge, and the graph represents executable artifact relationships rather than decoration.

## Phase 3 — Parametric editing, prompt interpretation, and branching

**Status: partial**

Outcome: users can control architecture directly and create non-destructive design lineages.

Delivered baseline:

- Constrained keyword parser for floor range, proportion, twist, terraces, courtyard, and material.
- Additive refinement field that regenerates deterministic geometry.

Remaining gate work:

- Runtime-validated canonical `BuildingSpec`, `Artifact`, `GenerationJob`, `DesignNode`, and `DesignRevision` contracts plus persisted-schema migration.
- Contextual parameter inspector for dimensions, floor count/height, volumes, setbacks, rotation, taper, facade, roof, and materials.
- Explicitly constrained local-language capability copy.
- Optional server-only Anthropic interpreter producing validated specifications; app remains complete without it.
- Non-destructive variation command: create child spec/artifact/node, retain source, render lineage, branch twice, and restore after reload.
- Undo/redo or equivalent safe recovery for graph/spec edits.

Exit gate: direct parameter edit and prompt refinement both produce observable child geometry while the parent remains available; two branches survive reload.

## Phase 4 — Hosted pretrained generation

**Status: partial — server create adapter only; live behavior unverified**

Outcome: configured users can generate hosted GLB assets safely without vendor coupling or secret exposure.

Delivered baseline:

- Server-only Meshy v2 preview request adapter and provider-status endpoint.
- Environment gating, request validation, server-only secret handling, and honest unverified status.

Remaining gate work:

- Protect paid endpoint against unauthorized use and require explicit provider/credit confirmation.
- Provider-neutral job adapter for create, retrieve/stream, supported cancellation, timeout, rate limit, and normalized errors.
- Persist task ID and status across reload; show real progress and recovery actions.
- Ingest completed GLB into the shared viewer and IndexedDB asset store before signed URL expiry.
- Add mocked lifecycle tests from documented contracts.
- With a supplied paid key, run and document a live preview/refine smoke test; otherwise retain “unverified.”

Exit gate: mocked acceptance passes and the UI never fabricates success. Live verification is separately recorded and is not inferred from mocks.

## Phase 5 — Rendering, artifacts, and persistence

**Status: partial**

Outcome: model, render, and project outputs are durable, reproducible, and presentation-ready.

Delivered baseline:

- Browser project save/load through an IndexedDB adapter.
- Current-view PNG path and procedural GLB export.

Remaining gate work:

- Artifact store for immutable spec/GLB/PNG outputs, without localStorage blobs.
- Render nodes bound to model ID, camera, material, lighting, background, and resolution.
- Camera presets; clay/shaded/wireframe materials; studio/daylight/sunset lighting; grid/axes/shadow toggles.
- 1024×1024 and 1600×900 exports, with guarded 1920×1080 support.
- Restore complete graph viewport, nodes, positions, settings, lineage, artifacts, and jobs.
- Validate downloaded GLB in an independent compatible loader/viewer.

Exit gate: render, export, refresh, and reopen acceptance tests pass for the bundled Terraced Tower Study and its two variations.

## Phase 6 — QA, resilience, and portfolio polish

**Status: partial**

Outcome: Sift is credible for portfolio demonstration and design-partner use.

- Add Playwright journeys for project → generate → edit → branch → render → save → reload → export.
- Add component tests for loading, empty, invalid geometry, persistence failure, missing credential, rate limit, timeout, and provider failure states.
- Add WebGL error boundary, informative unsupported-browser fallback, GPU cleanup checks, performance budget, and throttled expensive updates.
- Complete keyboard/touch accessibility and responsive audits; capture key desktop/focus/mobile screenshots.
- Finish first-run experience, Terraced Tower Study with two variations/render, and two distinct additional samples.
- Expand README with architecture, generation behavior, credentials, tests, deployment, independent-project attribution, and honest limitations.
- Add deploy/rollback/provider-cost runbook.

Exit gate: the ten acceptance scenarios in `PRODUCT_REQUIREMENTS.md` pass, no critical accessibility issue remains, production build/audit are clean, and `STATUS.md` identifies no incomplete MVP requirement.

## Post-MVP sequence

Only after Phase 6 exits:

1. Supabase project/asset sync.
2. Tripo or another second provider.
3. GLB import/reference context.
4. Advanced environments, comparison views, and rendering controls.

Do not expand into excluded video, generalized image tooling, BIM, model training, collaboration, payments, or plugin ecosystems while an MVP gate remains open.


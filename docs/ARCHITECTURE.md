# Architecture and data contracts

This document describes what is **implemented** (updated at P0.21). Open work is tracked in `STATUS.md` and `TASKS.md`; decisions and their reasons are in `DECISIONS.md`.

## System shape

```text
Browser (Next.js App Router, client-rendered studio)
  React Flow board ─┐
  Inspector ────────┼─▶ src/lib/workflow.ts   pure domain: evaluate, connect, run, edit, branch, snapshot, render, restore
  3D viewer (R3F) ──┘   src/lib/geometry.ts   BuildingSpec → per-floor slabs         (no rendering dependencies)
        │               src/lib/typologies.ts prompt → BuildingSpec (offline "local interpreter")
        │               src/lib/contracts.ts  Zod schemas — the compatibility boundary (schema v2)
        ├─ src/lib/storage.ts ─▶ IndexedDB: `sift-projects` (project records) + `sift-assets` (render PNGs, hosted GLBs)
        └─ /api/generate/* ──▶ src/lib/providers/* (server only): Meshy adapter, access-code + rate guard

Legacy PyTorch prototype (repo root): preserved for provenance, disconnected from the product.
```

The browser owns project state and procedural generation. Next.js route handlers are the **only** code that may read provider credentials or call paid APIs. UI components never touch IndexedDB or vendor payloads directly.

## Source map

| Path | Responsibility |
| --- | --- |
| `src/lib/contracts.ts` | Zod schemas: `BuildingSpec`, `Artifact`, `GenerationJob`, `DesignNode/Edge`, `DesignRevision`, `SiftProjectV2` (+ legacy v1 for migration) |
| `src/lib/graph.ts` | Port kinds per node type; `validateConnection` (type, duplicate input, cycles) |
| `src/lib/workflow.ts` | `evaluateGraph`, `runGeneration`, `editNodeGeometry`, `branchFrom`, `commitVariations`, `recordRender`, `versionsOf`, `restoreVersion`, `previewSpec` |
| `src/lib/spec-edit.ts` | Serializable `SpecEdit` operations validated through `buildingSpecSchema` + complexity budget |
| `src/lib/geometry.ts`, `three-building.ts`, `viewer.ts` | Layout, mesh building/disposal, camera presets and framing |
| `src/lib/render-settings.ts`, `render-image.ts` | Render settings/keys/resolution support; offscreen PNG renderer |
| `src/lib/limits.ts` | Triangle/mesh budgets |
| `src/lib/migrate.ts`, `backup.ts` | v1→v2 migration, store reconciliation; JSON backup/import |
| `src/lib/hosted.ts`, `hosted-client.ts` | Provider-neutral job state machine (pure); browser API wrapper |
| `src/lib/providers/` | **Server-only**: `meshy.ts` adapter, `guard.ts`, `hosted-http.ts` |
| `src/lib/storage.ts` | The only IndexedDB dependency: projects, assets, tombstones, probe |
| `src/lib/samples.ts`, `projects.ts` | Bundled samples (built from domain functions), project helpers |
| `src/components/studio/` | `studio-shell` (state + orchestration), `studio-node`, `inspector`, `model-preview`, `dashboard`, `paid-confirm`, `viewer-fallback` |

## Project contract (schema v2)

```ts
type SiftProjectV2 = {
  schemaVersion: 2; id; name; createdAt; updatedAt;
  viewport: { x; y; zoom };
  graph: { nodes: DesignNode[]; edges: DesignEdge[] };
  artifacts: Record<string, Artifact>;
  jobs: Record<string, GenerationJob>;
  revisions: Record<string, DesignRevision>;
  settings: { provider: "procedural" | "meshy"; viewer?: ViewerSettings };
};
```

Persisted records are parsed at the storage boundary. `projects-v1` (legacy) is read-only, migrated losslessly on read, and never deleted; deletions are recorded as tombstones. Unreadable records are written back untouched, never dropped. Any further schema change needs a migration before bumping the version.

### Building specification

`BuildingSpec` is the single source of procedural geometry truth: rectangle/circle footprint; volumes (`podium|tower|wing|core`) with floor ranges, scale, offsets, total twist, taper, setbacks; facade; roof; materials. Cross-field rules (unique volume ids, valid material refs, paired setback fields, floor/complexity ceilings) live in the schema. `computeLayout` expands it to slabs; `layoutComplexity` enforces the mesh budget.

### Artifacts, jobs, revisions

- **Artifacts are immutable**: `building-spec` (spec + brief + origin in `metadata`), `render-png` (settings, size, `inputKey`), `model-glb` (hosted mesh, `verified:false`). Binary payloads live in `sift-assets` under `Artifact.storageKey` (`asset:<id>`), not in the project record.
- **Jobs** (`GenerationJob`) hold execution state (queued/running/completed/failed/cancelled/timed-out/rate-limited), provider task id, progress, error, and optional result artifact — separate from artifacts.
- **Revisions** link a parent artifact to a child (`parameters`, `prompt`, `provider-regeneration`) with the user-visible instruction. Id collisions can never overwrite an existing artifact (generators are checked).

### Typed canvas

Node types: `prompt`, `generation`, `variation`, `model`, `render`. Ports carry kinds; `evaluateGraph` computes each node's output purely from upstream data. A Generation node produces output only after *Run* (it owns a spec artifact); Variation nodes derive their spec from follow-up text and stored `SpecEdit`s and are snapshotted into artifacts + revisions on save; Render nodes are `pending` until a render matching the current model + settings exists (`renderInputKey`).

## Generation contract

- `procedural`: runs in the browser (`runGeneration`); no network.
- `meshy` (optional, **unverified live**): `POST /api/generate` (needs `MESHY_ENABLED`, `MESHY_API_KEY`, `MESHY_ACCESS_CODE`; header `x-sift-access-code`; body `confirmSpend: true`), `GET|DELETE /api/generate/{taskId}` (normalized status / cancel — Meshy only deletes queued tasks), `GET /api/generate/{taskId}/model` (fresh task lookup, host-allowlisted, size-capped GLB download). The browser polls with backoff, resumes after reload (re-entering the access code), and stores the GLB locally rather than relying on expiring signed URLs.

## Security and privacy

- `MESHY_API_KEY` is server-only; it is never serialized, logged, returned, or stored client-side. Hosted calls require the access code (constant-time compare), explicit confirmation, and pass per-IP/daily limits (per server instance).
- Signed asset URLs never reach the browser; only `*.meshy.ai` HTTPS URLs are fetched server-side.
- Prompts and models are user content; there is no analytics capture. Project data stays in the visitor's browser.

## Export and viewing

PNG: client-side capture of the live canvas, or an offscreen Render node at an exact resolution. GLB: built from the same slab geometry as the viewer (shaded materials, independent of viewer mode), includes a ground plate. Hosted GLBs are shown with `GLTFLoader` (triangle-capped) and can be downloaded.

## Failure handling

Unsupported WebGL, context loss, render errors, over-complex models, unavailable/failed storage, missing credentials, and provider failures each have an explicit UI state with a recovery path (see `STATUS.md`). Async work never applies its result to a different project than the one it started on.

## Testing strategy

- **Unit/domain** (Vitest, node): contracts, migration, graph, geometry, workflow, edits, render settings, hosted jobs, limits, backup, samples, persistence (fake-indexeddb), Meshy adapter + routes (mocked fetch).
- **Component** (Vitest + jsdom): dashboard, inspector, paid-confirm dialog, viewer fallback/boundary.
- **End-to-end** (Playwright, production build): the ten acceptance scenarios, the featured sample, four responsive widths, axe WCAG A/AA scans; screenshots in `docs/evidence/`.
- **Provider**: contract tests use mocked HTTP; live smoke tests are manual and opt-in because they consume credits.

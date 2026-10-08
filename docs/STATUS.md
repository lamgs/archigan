# Current status

**Updated:** 2026-10-08  
**Branch:** `claude/determined-sagan-v8sy6v` (uncommitted P0.09 work)  
**Milestone:** Phase 1 baseline complete; master-brief MVP gap closure active  
**Overall:** Working local vertical slice, not yet MVP-complete

## Deployment

- Vercel project: `booth-os/archigan`.
- Recorded URL: `https://archigan-ctjx7uh6t-booth-os.vercel.app`.
- 2026-10-09: all deployments from `f43fc44` through P0.12 failed on Vercel; adding `vercel.json` (framework `nextjs`) made commit `6fa7198` deploy successfully per the GitHub `Vercel` status. Logs/runtime remain unverified.
- Observed 2026-10-08: hostname responds with a `302` redirect to Vercel SSO/Deployment Protection.
- Deployment metadata and authenticated application smoke test remain unverified because this session lacks `booth-os` connector/CLI authorization. See `DEPLOYMENT.md`.

## Blockers and unverified items

Record anything that stops or limits work here (with the date and what would unblock it). Remove an entry only when it is resolved.

| Date | Blocker | Effect | What would unblock it |
| --- | --- | --- | --- |
| 2026-10-09 | `docs.meshy.ai` is blocked by the agent network proxy (WebFetch `EGRESS_BLOCKED`); only search snippets were readable. | The Meshy adapter follows the publicly documented contract as summarized by search (statuses, endpoints, DELETE 409 on running tasks); exact response field names (`model_urls`, `task_error`, `expires_at`) are unconfirmed, so parsing is lenient and everything is labelled **unverified**. | Allow `docs.meshy.ai` through the proxy, or paste the Text-to-3D reference; or run one real task and compare the JSON. |
| 2026-10-09 | No Meshy API key/account in this environment. | No live hosted call has ever been made. Only mocked documented-contract tests exist; `verified` is hard-wired `false`. | A paid Meshy key set as `MESHY_API_KEY` (+ `MESHY_ENABLED=true`, `MESHY_ACCESS_CODE`) in a trusted environment, then the manual smoke test in `DEPLOYMENT.md`. |
| 2026-10-09 | All WebGL checks ran in headless Chromium on SwiftShader (software rendering) in this container. | Real-GPU frame rates, memory pressure, mobile GPUs, Safari/Firefox, and the `deviceMemory` heuristic for 1920×1080 are untested. | Manual pass on real desktop + mobile devices/browsers; record results here. |
| 2026-10-09 | `vercel.com` / `api.vercel.com` are unreachable and no Vercel token is available. | Build logs and the deployed app (behind Deployment Protection) cannot be inspected by the agent; deploy health is inferred from GitHub commit statuses only. | An authorized Vercel identity (`vercel inspect <id> --logs`), or the user pasting logs/errors. |

## Working now

- Canonical v2 contracts (`BuildingSpec`, `Artifact`, `GenerationJob`, `DesignNode`/ports, `DesignRevision`, `SiftProjectV2`), graph connection/cycle validation, and a lossless v1→v2 migration. Storage writes v2 (`projects-v2`), reads legacy `projects-v1` read-only, preserves unreadable records, and saves in one atomic transaction.
- Geometry engine: `computeLayout(BuildingSpec)` (rectangle/circle footprints, multi-volume podium/tower, offsets, twist, taper, setbacks, roof, glazing, bounds, clamping warnings) rendered by `three-building.ts`; `deriveBuildingSpec` maps prompts to five typologies. Preview, GLB export, and the massing node now use `BuildingSpec`; browser-verified distinct silhouettes for terraced, twin, cylindrical, and rotated briefs.
- Project dashboard (`dashboard.tsx`, `projects.ts`): first-run empty state with example briefs, new/open/rename/delete with confirmation, samples open as editable copies, autosave on Generate. Storage CRUD is tested against `fake-indexeddb`.
- Typed executable canvas: prompt/generation/variation/model/render nodes with typed ports, validated wiring (type mismatch, duplicate input, cycles rejected), add toolbar and contextual “Add next”, Run creates an immutable building-spec artifact + job (+ revision link on re-run), stale detection, persisted viewport. App state is now `SiftProjectV2` end to end; v1 exists only for migration (`legacy-fixtures.ts`).
- Contextual inspector: provider choice (Generation node), full geometry controls (footprint, floor height, per-volume floors/scale/offset/twist/taper/setbacks, facade, roof, materials) with validated edits; edits on a Generation node make a new artifact + revision, edits on a Variation node are stored on that node; nothing overwrites the source artifact.
- Non-destructive branching: Branch action, labelled lanes (Branch A/B/…), per-branch artifacts + revisions with the shared parent, version list in the inspector, “Use this version” on Generation nodes. Browser-verified: two branches with different geometry stay distinct and restore identically after reload.
- Expanded viewer: focus mode, bounds-based framing, five camera presets (three true orthographic), display modes, grid/axes/shadow toggles, keyboard orbit/zoom/frame; browser-verified, including that drags/wheel inside the viewer (inline and focus) never move the outer canvas.
- Render pipeline: Render nodes render offscreen at exact pixel sizes, persist PNG assets outside the project record, track freshness, offer download/preview in the inspector; deleting a project deletes its render assets.
- Persistence: everything (graph, viewport, prompts, artifacts/specs, revisions, jobs, project + viewer settings, render PNG assets) is stored in browser IndexedDB only; autosave with a visible Saved/Unsaved/Saving/error badge; verified by a full-board restore test and a browser refresh test. Data is local to each browser profile and origin (no cloud sync until P1.01).
- Hosted (Meshy) lifecycle — **unverified against a live account**: fail-closed paid-request guard, confirmation dialog naming Meshy, create/poll/cancel with normalized states, reload-resume (re-enter access code), GLB ingestion into IndexedDB, hosted model viewing/download. Verified only with mocked contract tests and a browser run against a mocked API.
- Robustness: unsupported-WebGL, context-lost, render-error, too-complex, storage-unavailable, save-failed, missing-credential, and provider-failure states all have explicit UI with recovery paths (restart view, retry save, download backup/import backup). GPU resources are disposed (unit-tested) and the viewer idles at zero frames.
- Acceptance automation: 10 Playwright journeys (one per PRD acceptance scenario, incl. axe WCAG A/AA scans) against the production build + 17 jsdom component tests; screenshots retained in `docs/evidence/`. Run with `npm run test:e2e`.
- Shared Codex/Claude operating docs and explicit product boundary.
- Next.js/TypeScript app shell with React Flow workflow canvas.
- Deterministic procedural architectural massing in R3F.
- Prompt refinement, sample projects, IndexedDB save/load.
- Client-side PNG and GLB export paths.
- Server-only Meshy adapter boundary and configuration status.
- Browser-verified prompt/refinement generation, IndexedDB save, 3D rendering, camera controls, and GLB export completion.

## Known limitations

- Meshy has not been called with a real account; status must remain “unverified.” Hosted results are fixed meshes (not editable); variation/render nodes need the Local provider’s parametric spec. The access code is held in memory only, so after a reload it must be re-entered to resume polling. Rate limits are per server instance (in-memory).
- Render nodes have no behavior until P0.16; `/api/generate` still returns legacy `MassingSpec` and is unused by the UI; variation nodes derive their spec live (snapshotted for lineage only) (not yet persisted as child artifacts — P0.14).
- Storage logic is covered via pure functions (`reconcileStores`); IndexedDB itself is not exercised by automated tests (no fake-indexeddb yet).
- Variation output is derived live from prompt+refinement (not persisted as an artifact, not directly parameter-editable until P0.13/P0.14). Courtyard voids are not modeled; vertical/grid facades render as ribbon glazing; the legacy `three-massing.ts` builder is now unused.
- Direct parameter controls, contextual inspector, non-destructive design branches, and restored lineage are not implemented.
- The viewer’s own “PNG” button still captures the live canvas at its on-screen size (render nodes are the resolution-specific path); live viewer settings are not persisted; renders are produced on the main thread and block briefly at large sizes.
- Orphaned render assets (superseded renders) are kept until the project is deleted; deleted projects are not recoverable; a blank new project is not persisted until its first Generate (schema requires a prompt).
- Meshy create code exists, but the client job lifecycle, polling/streaming, persistent GLB ingestion, paid-request protection, and provider-mocked tests remain open.
- Procedural output remains conceptual massing, not BIM, code-compliant, structural, or fabrication geometry.
- E2E runs only in Chromium on software rendering (SwiftShader); no Firefox/WebKit/mobile or real-GPU coverage; no CI workflow is committed (it would need workflow-scope push permission).
- The Vercel deployment is access-protected and has not been smoke-tested behind protection from this session.
- The legacy Python prototype remains at the root until a later cleanup decision.

## Verification

- `npm run test`: passed, 17 files / 180 unit+component tests; 10 Playwright e2e tests.
- `npm run lint`: passed with zero warnings.
- `npm run build`: passed on Next.js 16.4.0; `/`, `/api/generate`, and `/api/providers` built successfully.
- `npm audit --audit-level=high`: passed, zero known vulnerabilities.
- Manual browser smoke test: passed for generation, refinement, save, WebGL rendering, camera reset, and GLB serialization/download trigger.

## Next action

Implement P0.21 (portfolio completion: Terraced Tower Study sample with two saved variations + a render and two distinct typology presets; first-run copy; responsive/accessibility review; README architecture/setup/credentials/tests/deploy/limitations/independence sections). Then the P0 acceptance gate review (all P0 boxes, all ten scenarios, live-provider status honestly reported), which move the UI onto v2 and persist the `BuildingSpec` artifact so P0.13 inspector edits create revisions. Do not start post-MVP Supabase/Tripo work while P0 remains open.


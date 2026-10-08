# Current status

**Updated:** 2026-10-08  
**Branch:** `claude/determined-sagan-v8sy6v` (uncommitted P0.09 work)  
**Milestone:** Phase 1 baseline complete; master-brief MVP gap closure active  
**Overall:** Working local vertical slice, not yet MVP-complete

## Deployment

- Vercel project: `booth-os/archigan`.
- Recorded URL: `https://archigan-ctjx7uh6t-booth-os.vercel.app`.
- Observed 2026-10-08: hostname responds with a `302` redirect to Vercel SSO/Deployment Protection.
- Deployment metadata and authenticated application smoke test remain unverified because this session lacks `booth-os` connector/CLI authorization. See `DEPLOYMENT.md`.

## Working now

- Canonical v2 contracts (`BuildingSpec`, `Artifact`, `GenerationJob`, `DesignNode`/ports, `DesignRevision`, `SiftProjectV2`), graph connection/cycle validation, and a lossless v1→v2 migration. Storage writes v2 (`projects-v2`), reads legacy `projects-v1` read-only, preserves unreadable records, and saves in one atomic transaction.
- Geometry engine: `computeLayout(BuildingSpec)` (rectangle/circle footprints, multi-volume podium/tower, offsets, twist, taper, setbacks, roof, glazing, bounds, clamping warnings) rendered by `three-building.ts`; `deriveBuildingSpec` maps prompts to five typologies. Preview, GLB export, and the massing node now use `BuildingSpec`; browser-verified distinct silhouettes for terraced, twin, cylindrical, and rotated briefs.
- Shared Codex/Claude operating docs and explicit product boundary.
- Next.js/TypeScript app shell with React Flow workflow canvas.
- Deterministic procedural architectural massing in R3F.
- Prompt refinement, sample projects, IndexedDB save/load.
- Client-side PNG and GLB export paths.
- Server-only Meshy adapter boundary and configuration status.
- Browser-verified prompt/refinement generation, IndexedDB save, 3D rendering, camera controls, and GLB export completion.

## Known limitations

- Meshy has not been called with a real account; status must remain “unverified.”
- The UI still renders the v1 projection (`toLegacyProject`); typed ports/validation exist in the domain layer but are not wired into React Flow `onConnect`, and the viewport is not yet restored.
- Storage logic is covered via pure functions (`reconcileStores`); IndexedDB itself is not exercised by automated tests (no fake-indexeddb yet).
- Persisted projects still store legacy `MassingSpec`; the viewer derives `BuildingSpec` from prompt+refinement at render time (not persisted, not user-editable until P0.13/P0.17). Courtyard voids are not modeled; vertical/grid facades render as ribbon glazing; the legacy `three-massing.ts` builder is now unused.
- Direct parameter controls, contextual inspector, non-destructive design branches, and restored lineage are not implemented.
- Camera presets, expanded viewer, selectable render/material/lighting modes, render nodes, and resolution-specific PNG artifacts are not implemented.
- Project create/delete dashboard flows and complete asset/job persistence are not implemented.
- Meshy create code exists, but the client job lifecycle, polling/streaming, persistent GLB ingestion, paid-request protection, and provider-mocked tests remain open.
- Procedural output remains conceptual massing, not BIM, code-compliant, structural, or fabrication geometry.
- Automated browser end-to-end coverage is not yet committed; the current flow has been manually smoke-tested in the in-app browser.
- The Vercel deployment is access-protected and has not been smoke-tested behind protection from this session.
- The legacy Python prototype remains at the root until a later cleanup decision.

## Verification

- `npm run test`: passed, 4 files / 34 tests.
- `npm run lint`: passed with zero warnings.
- `npm run build`: passed on Next.js 16.4.0; `/`, `/api/generate`, and `/api/providers` built successfully.
- `npm audit --audit-level=high`: passed, zero known vulnerabilities.
- Manual browser smoke test: passed for generation, refinement, save, WebGL rendering, camera reset, and GLB serialization/download trigger.

## Next action

Implement P0.11 (project dashboard/CRUD), then P0.12–P0.14, which move the UI onto v2 and persist the `BuildingSpec` artifact so P0.13 inspector edits create revisions. Do not start post-MVP Supabase/Tripo work while P0 remains open.


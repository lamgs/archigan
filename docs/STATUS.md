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

## Working now

- Canonical v2 contracts (`BuildingSpec`, `Artifact`, `GenerationJob`, `DesignNode`/ports, `DesignRevision`, `SiftProjectV2`), graph connection/cycle validation, and a lossless v1→v2 migration. Storage writes v2 (`projects-v2`), reads legacy `projects-v1` read-only, preserves unreadable records, and saves in one atomic transaction.
- Geometry engine: `computeLayout(BuildingSpec)` (rectangle/circle footprints, multi-volume podium/tower, offsets, twist, taper, setbacks, roof, glazing, bounds, clamping warnings) rendered by `three-building.ts`; `deriveBuildingSpec` maps prompts to five typologies. Preview, GLB export, and the massing node now use `BuildingSpec`; browser-verified distinct silhouettes for terraced, twin, cylindrical, and rotated briefs.
- Project dashboard (`dashboard.tsx`, `projects.ts`): first-run empty state with example briefs, new/open/rename/delete with confirmation, samples open as editable copies, autosave on Generate. Storage CRUD is tested against `fake-indexeddb`.
- Typed executable canvas: prompt/generation/variation/model/render nodes with typed ports, validated wiring (type mismatch, duplicate input, cycles rejected), add toolbar and contextual “Add next”, Run creates an immutable building-spec artifact + job (+ revision link on re-run), stale detection, persisted viewport. App state is now `SiftProjectV2` end to end; v1 exists only for migration (`legacy-fixtures.ts`).
- Contextual inspector: provider choice (Generation node), full geometry controls (footprint, floor height, per-volume floors/scale/offset/twist/taper/setbacks, facade, roof, materials) with validated edits; edits on a Generation node make a new artifact + revision, edits on a Variation node are stored on that node; nothing overwrites the source artifact.
- Non-destructive branching: Branch action, labelled lanes (Branch A/B/…), per-branch artifacts + revisions with the shared parent, version list in the inspector, “Use this version” on Generation nodes. Browser-verified: two branches with different geometry stay distinct and restore identically after reload.
- Expanded viewer: focus mode, bounds-based framing, five camera presets (three true orthographic), display modes, grid/axes/shadow toggles, keyboard orbit/zoom/frame; browser-verified, including that drags/wheel inside the viewer (inline and focus) never move the outer canvas.
- Shared Codex/Claude operating docs and explicit product boundary.
- Next.js/TypeScript app shell with React Flow workflow canvas.
- Deterministic procedural architectural massing in R3F.
- Prompt refinement, sample projects, IndexedDB save/load.
- Client-side PNG and GLB export paths.
- Server-only Meshy adapter boundary and configuration status.
- Browser-verified prompt/refinement generation, IndexedDB save, 3D rendering, camera controls, and GLB export completion.

## Known limitations

- Meshy has not been called with a real account; status must remain “unverified.”
- Render nodes have no behavior until P0.16; `/api/generate` still returns legacy `MassingSpec` and is unused by the UI; variation nodes derive their spec live (snapshotted for lineage only) (not yet persisted as child artifacts — P0.14).
- Storage logic is covered via pure functions (`reconcileStores`); IndexedDB itself is not exercised by automated tests (no fake-indexeddb yet).
- Variation output is derived live from prompt+refinement (not persisted as an artifact, not directly parameter-editable until P0.13/P0.14). Courtyard voids are not modeled; vertical/grid facades render as ribbon glazing; the legacy `three-massing.ts` builder is now unused.
- Direct parameter controls, contextual inspector, non-destructive design branches, and restored lineage are not implemented.
- Render nodes and resolution-specific PNG artifacts are not implemented (PNG export still captures the live canvas at its on-screen size); lighting/background controls do not exist yet; viewer settings are not persisted per node/project.
- Complete asset/job persistence is not implemented; deleted projects are not recoverable; a blank new project is not persisted until its first Generate (schema requires a prompt).
- Meshy create code exists, but the client job lifecycle, polling/streaming, persistent GLB ingestion, paid-request protection, and provider-mocked tests remain open.
- Procedural output remains conceptual massing, not BIM, code-compliant, structural, or fabrication geometry.
- Automated browser end-to-end coverage is not yet committed; the current flow has been manually smoke-tested in the in-app browser.
- The Vercel deployment is access-protected and has not been smoke-tested behind protection from this session.
- The legacy Python prototype remains at the root until a later cleanup decision.

## Verification

- `npm run test`: passed, 9 files / 85 tests.
- `npm run lint`: passed with zero warnings.
- `npm run build`: passed on Next.js 16.4.0; `/`, `/api/generate`, and `/api/providers` built successfully.
- `npm audit --audit-level=high`: passed, zero known vulnerabilities.
- Manual browser smoke test: passed for generation, refinement, save, WebGL rendering, camera reset, and GLB serialization/download trigger.

## Next action

Implement P0.16 (render artifact pipeline: render nodes bound to model, camera preset, view mode, light, background, resolution; PNG artifacts persisted), then P0.17, which move the UI onto v2 and persist the `BuildingSpec` artifact so P0.13 inspector edits create revisions. Do not start post-MVP Supabase/Tripo work while P0 remains open.


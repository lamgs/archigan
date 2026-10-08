# Current status

**Updated:** 2026-10-08  
**Branch:** `main` (tracking `origin/main`)  
**Milestone:** Phase 1 baseline complete; master-brief MVP gap closure active  
**Overall:** Working local vertical slice, not yet MVP-complete

## Deployment

- Vercel project: `booth-os/archigan`.
- Recorded URL: `https://archigan-ctjx7uh6t-booth-os.vercel.app`.
- Observed 2026-10-08: hostname responds with a `302` redirect to Vercel SSO/Deployment Protection.
- Deployment metadata and authenticated application smoke test remain unverified because this session lacks `booth-os` connector/CLI authorization. See `DEPLOYMENT.md`.

## Working now

- Shared Codex/Claude operating docs and explicit product boundary.
- Next.js/TypeScript app shell with React Flow workflow canvas.
- Deterministic procedural architectural massing in R3F.
- Prompt refinement, sample projects, IndexedDB save/load.
- Client-side PNG and GLB export paths.
- Server-only Meshy adapter boundary and configuration status.
- Browser-verified prompt/refinement generation, IndexedDB save, 3D rendering, camera controls, and GLB export completion.

## Known limitations

- Meshy has not been called with a real account; status must remain “unverified.”
- The current graph is a guided lifecycle without typed ports, connection validation, project viewport restore, or executable artifact semantics.
- Current `MassingSpec` generates stacked-box studies; the canonical multi-volume `BuildingSpec` and three distinct typologies are not implemented.
- Direct parameter controls, contextual inspector, non-destructive design branches, and restored lineage are not implemented.
- Camera presets, expanded viewer, selectable render/material/lighting modes, render nodes, and resolution-specific PNG artifacts are not implemented.
- Project create/delete dashboard flows and complete asset/job persistence are not implemented.
- Meshy create code exists, but the client job lifecycle, polling/streaming, persistent GLB ingestion, paid-request protection, and provider-mocked tests remain open.
- Procedural output remains conceptual massing, not BIM, code-compliant, structural, or fabrication geometry.
- Automated browser end-to-end coverage is not yet committed; the current flow has been manually smoke-tested in the in-app browser.
- The Vercel deployment is access-protected and has not been smoke-tested behind protection from this session.
- The legacy Python prototype remains at the root until a later cleanup decision.

## Verification

- `npm run test`: passed, 2 files / 6 tests.
- `npm run lint`: passed with zero warnings.
- `npm run build`: passed on Next.js 16.4.0; `/`, `/api/generate`, and `/api/providers` built successfully.
- `npm audit --audit-level=high`: passed, zero known vulnerabilities.
- Manual browser smoke test: passed for generation, refinement, save, WebGL rendering, camera reset, and GLB serialization/download trigger.

## Next action

Implement `P0.09 Canonical domain contracts and migration`, then `P0.10 Architectural geometry engine`. These contracts unblock branching, typed canvas execution, persistence, rendering, and provider jobs. Do not start post-MVP Supabase/Tripo work while P0 remains open.


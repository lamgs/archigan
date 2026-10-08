# Current status

**Updated:** 2026-10-08 (Tripo adapter migrated to the current official v3 contract)
**Branch:** `claude/intelligent-ritchie-3zzcsl` (based on `claude/determined-sagan-v8sy6v`) (work after merged PR #1; see git for clean/dirty state)  
**Milestone:** All P0 tasks implemented; **P0 gate review completed — PASS with documented exceptions** (below)  
**Overall:** Complete local-first MVP candidate. **By owner decision (2026-10-09), exceptions E1–E4 remain OUTSTANDING and the MVP is NOT declared complete.** Do not mark it complete, and do not start P1 work, until the owner says otherwise.

## P0 gate review (2026-10-09, commit `a76ad35`)

Verified from a **fresh clone of the remote branch**: `npm ci` → lint clean → `tsc` clean → 194 unit/component tests → production build → `npm audit` 0 vulnerabilities → 25 Playwright tests passed on two consecutive runs; Vercel commit status `success`; working tree clean afterwards.

| Gate criterion (`IMPLEMENTATION_PLAN.md` Phase 6 / `PRODUCT_REQUIREMENTS.md`) | Result | Evidence |
| --- | --- | --- |
| All 21 P0 tasks checked | Pass | `TASKS.md` |
| Acceptance scenarios 1–8 | Pass | `e2e/acceptance.spec.ts` + `docs/evidence/` |
| Scenario 9 (hosted jobs) | Pass **as mocked contract**; live reported separately | E1 |
| Scenario 10 (TS, lint, tests, Playwright, build, a11y, screenshots) | Pass | axe: 0 violations of any impact (incl. best-practice) on dashboard/studio/inspector/render/focus at 1280 and 390 px |
| Principal sample matches the brief (12 stories, 4-floor podium + 8-floor tower, setbacks every 2 floors, glazed) | Pass (fixed in this review) | `samples.test.ts` |
| Viewer ≈ render, GLB valid | Pass (fixed/added in this review) | tone-mapping parity test; Khronos glTF-Validator: 0 errors |
| No secrets in client bundle / history; hosted endpoints fail closed | Pass | `e2e/security.spec.ts`, history scan |
| Undo/redo or equivalent safe recovery | Pass (added in this review) | `e2e/undo.spec.ts`, ADR-013 |
| Deploy/rollback/provider-cost runbook, performance budget | Pass (added) | `DEPLOYMENT.md` |

**Defects found and fixed by the review:** product descriptor read “Architectural intelligence” (now “AI Architectural Form Studio”); the principal sample did not match its specified brief because the interpreter ignored numbers; renders used different tone mapping than the viewer; no undo/redo for graph edits; heading-order and ARIA-role accessibility issues; evidence screenshots churned on every test run (now opt-in via `E2E_EVIDENCE=1`); generated `__pycache__` files were tracked (ADR-015).

**Exceptions the owner must accept before declaring the MVP complete** (each also appears in the blockers table below):

- **E1 — Live Meshy is unverified.** No real-account call has ever been made; `verified` is hard-wired `false`. The PRD only requires mock/live to be labelled and reported separately, which they are.
- **E2 — Real-GPU, Safari/Firefox and mobile-device testing was not done.** Everything ran in headless Chromium on software rendering. Touch interaction was not tested on a touch device (only responsive layouts).
- **E3 — The deployed app behind Vercel Deployment Protection was not inspected;** deployment health is the GitHub commit status only.
- **E4 — Minor deviations from the brief:** the “Add” toolbar is a horizontal pill at the top-left of the board rather than a vertical left toolbar; the optional server-side LLM interpreter was not built (ADR-014).

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
| 2026-10-08 | `docs.meshy.ai` was rechecked directly with `curl`; the Envoy CONNECT tunnel returned `403 Forbidden`. | The Meshy adapter still follows search-snippet summaries (statuses, endpoints, DELETE 409 on running tasks); exact response field names (`model_urls`, `task_error`, `expires_at`) remain unconfirmed, so parsing is lenient and everything is labelled **unverified**. | Allow `docs.meshy.ai` through the proxy, paste the Text-to-3D reference, or run one real task and compare the JSON. |
| 2026-10-08 | `fal.ai` and `docs.fal.ai` were rechecked directly with `curl`; both Envoy CONNECT tunnels returned `403 Forbidden`. Tencent's `www.tencentcloud.com/products/ai3d` page is blocked the same way. The official international Go SDK mirror on GitHub is reachable. | Tencent's native `ai3d/v20250513` SDK partially confirms **Pro upstream semantics**: prompt ≤1024 UTF-8 characters; `EnablePBR`; `FaceCount` default 500,000/range 40,000–1,500,000; `GenerateType`; `WAIT/RUN/FAIL/DONE`; and `ResultFile3Ds[{Type,Url,PreviewImageUrl}]`, with 24-hour task/file validity and three concurrent tasks by default. It does **not** document fal.ai's wrapper: Rapid, queue app-id vs full-path routes, snake_case option/output names (`model_glb`/`model_urls.glb`), errors, or cancel behavior remain unconfirmed. The fal adapter remains **UNVERIFIED** and lenient; prices remain estimates. | Allow the fal.ai documentation hosts through the proxy, paste the relevant references, or run one real task per fal provider and compare the JSON (DEPLOYMENT smoke test). |
| 2026-10-08 | `developers.tripo3d.ai/en/docs`, `docs.tripo3d.ai`, and `platform.tripo3d.ai` all return Envoy CONNECT `403 Forbidden`. Tripo's official JavaScript/TypeScript SDK (`@vastai/tripo-sdk` v0.3.0, `VAST-AI-Research/tripo-js-sdk`, commit `caa42b9`) is reachable through npm/GitHub. | The newer SDK explicitly targets v3 rather than the older `/v2/openapi/task` API. The adapter now uses global base URL `https://openapi.tripo3d.ai/v3`, Bearer auth, `POST /generation/text-to-model` with `{prompt}`, `GET /tasks/{id}`, the documented status vocabulary, and primary `output.model_url` (with legacy output fallbacks). It confirms code 2010 as insufficient credits and exposes no cancel operation. Prompt limits, asset-host guarantees, pricing, and other envelope-code meanings remain unconfirmed. No live call has been made, so Tripo remains **UNVERIFIED** and lenient. | Obtain the inaccessible reference pages for the remaining details, or run one authorized real-account smoke test and compare the raw JSON. |
| 2026-10-09 | No hosted-provider API key/account (Meshy, Tripo, fal.ai) in this environment. | No live hosted call has ever been made for any provider. Only mocked documented-contract tests exist; `verified` is hard-wired `false`. | A paid key per provider (`MESHY_API_KEY`/`TRIPO_API_KEY`/`FAL_KEY` + its `*_ENABLED=true` + `SIFT_ACCESS_CODE`) in a trusted environment, then the manual smoke test in `DEPLOYMENT.md`. |
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
- Multi-provider hosted lifecycle (Hunyuan3D Rapid/Pro via fal.ai, Tripo, Meshy; P1.02, ADR-017) — **all unverified against live accounts**: provider picker with configured/unverified/estimated-cost display, per-provider fail-closed config, shared access code + limits, confirmation naming the selected provider, Cancel only where supported (Tripo: stop waiting), polling by each job's own provider. Original Meshy lifecycle: fail-closed paid-request guard, confirmation dialog naming Meshy, create/poll/cancel with normalized states, reload-resume (re-enter access code), GLB ingestion into IndexedDB, hosted model viewing/download. Verified only with mocked contract tests and a browser run against a mocked API.
- Robustness: unsupported-WebGL, context-lost, render-error, too-complex, storage-unavailable, save-failed, missing-credential, and provider-failure states all have explicit UI with recovery paths (restart view, retry save, download backup/import backup). GPU resources are disposed (unit-tested) and the viewer idles at zero frames.
- Portfolio sample: *Terraced Tower Study* (two retained branches with lineage + self-rendering Render node), Twin Towers and Cylindrical Residence presets, first-run guidance with the interpreter's vocabulary, responsive layouts verified at 1280/1024/768/390 px with axe scans.
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
- Courtyard voids are not modeled; vertical/grid facades render as ribbon glazing; the legacy `three-massing.ts` builder is unused.
- The viewer’s own “PNG” button still captures the live canvas at its on-screen size (render nodes are the resolution-specific path); live viewer settings are not persisted; renders are produced on the main thread and block briefly at large sizes.
- Orphaned render assets (superseded renders) are kept until the project is deleted; deleted projects are not recoverable; a blank new project is not persisted until its first Generate (schema requires a prompt).
- Procedural output remains conceptual massing, not BIM, code-compliant, structural, or fabrication geometry.
- E2E runs only in Chromium on software rendering (SwiftShader); no Firefox/WebKit/mobile or real-GPU coverage; no CI workflow is committed (it would need workflow-scope push permission).
- The Vercel deployment is access-protected and has not been smoke-tested behind protection from this session.
- The legacy Python prototype remains at the root until a later cleanup decision.

## Verification

- `npm run test`: passed, 21 files / 308 unit+component+route tests; 28 Playwright e2e tests (scenario 9 now runs once per hosted provider). Verified 2026-10-08 from a fresh clone of the remote branch: `npm ci` → lint → tsc → vitest → production build → Playwright → `npm audit` (0 vulnerabilities). All mocked; no live provider call.
- `npm run lint`: passed with zero warnings.
- `npm run build`: passed on Next.js 16.4.0; `/`, `/api/generate`, and `/api/providers` built successfully.
- `npm audit --audit-level=high`: passed, zero known vulnerabilities.
- Manual browser smoke test: passed for generation, refinement, save, WebGL rendering, camera reset, and GLB serialization/download trigger.
- 2026-10-08 continuation verification at `fde122a` plus the build/test-harness compatibility changes: `npm ci` (with a writable `/tmp` cache), lint, `tsc --noEmit`, 308 Vitest tests, production build, and all 28 Playwright tests passed. The first system-Chromium Playwright run passed 27/28 and exposed a transformed-node hit-test timeout; after making the test-only node selector dispatch an explicit DOM click, the focused mutation check and final full suite passed.
- 2026-10-08 Tripo v3 migration: updated v3 regression expectations failed in four places against the old v2 adapter, then passed after the route/output/error mapping changes. Final lint, `tsc --noEmit`, 308 Vitest tests, production build, and all 28 Playwright tests passed. No live provider call was made.

## Next action (handoff to the next session)

**Owner decision recorded:** keep E1–E4 outstanding. Nothing is left to build for P0; remaining work is closing exceptions, each of which needs something this environment lacked. Pick up in this order, recording results in the blockers table and a new HANDOFF entry:

1. **E1 — live Meshy:** needs a throwaway low-credit `MESHY_API_KEY` in a trusted environment (never commit it). Run the six-step smoke test in `DEPLOYMENT.md`. Compare the real JSON against `normalizeTask` in `src/lib/providers/meshy.ts` (`model_urls.glb`, `task_error.message`, `expires_at` are unconfirmed) and adjust the adapter/tests. Only then consider changing the hard-wired `verified:false` in `meshyStatus()`.
2. **E3 — deployed app:** with an authorized Vercel identity run `vercel inspect <dpl> --logs`, or smoke-test with `E2E_BASE_URL=https://<host> VERCEL_AUTOMATION_BYPASS_SECRET=<secret> npx playwright test e2e/acceptance.spec.ts -g "1\.|8\."`. Also clear the project-level Output Directory override (`public`) in Vercel settings (`vercel.json` currently masks it).
3. **E2 — real devices/browsers:** manual pass on a real GPU, an iPhone/Android, Safari and Firefox (viewer, render PNGs at 1920×1080, touch orbit/pan, autosave, backup/import). Record findings; fix what breaks.
4. **E4 — deviations:** restyle the “Add” toolbar as a compact vertical left toolbar; optionally build the server-side LLM interpreter (see ADR-014) — only if the owner wants them.

Do not start P1.01 (Supabase sync) or P1.03 (GLB import) while any of the above is undecided. **Exception — owner-approved 2026-10-09: P1.02 multi-provider hosted generation (Hunyuan3D via fal.ai, Tripo, Meshy; ADR-016) may proceed.** It does not close E1–E4, and each new provider is added to the E1 “unverified live” list. **P1.02 is now implemented (mocked only); next for it is a real-account smoke test per provider (E1 list).** Latest verified counts are under Verification.

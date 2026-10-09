# Current status

**Updated:** 2026-10-09 (ADR-019: bloat removal)  
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

- **E1 — Live Tripo is unverified.** (Originally Meshy; hosted generation is Tripo only since ADR-018.) No real-account call has ever been made; `verified` is hard-wired `false`. The PRD only requires mock/live to be labelled and reported separately, which they are.
- **E2 — Real-GPU, Safari/Firefox and mobile-device testing was not done.** Everything ran in headless Chromium on software rendering. Touch interaction was not tested on a touch device (only responsive layouts).
- **E3 — The deployed app behind Vercel Deployment Protection was not inspected;** deployment health is the GitHub commit status only.
- **E4 — Minor deviations from the brief:** ~~the “Add” toolbar was a horizontal pill~~ — **resolved 2026-10-09**: it is now a vertical left toolbar; the optional server-side LLM interpreter was not built (ADR-014).

## Deployment

- Vercel project: `gabelam/archigan` (team is `gabelam`, owner-confirmed 2026-10-08; older notes saying `booth-os` are outdated).
- Production domain: `https://archigan.vercel.app` (stable; per-deployment URLs change every build and are not recorded).
- 2026-10-09: all deployments from `f43fc44` through P0.12 failed on Vercel; adding `vercel.json` (framework `nextjs`) made commit `6fa7198` deploy successfully per the GitHub `Vercel` status. Logs/runtime remain unverified.
- Observed 2026-10-08: hostname responds with a `302` redirect to Vercel SSO/Deployment Protection.
- Deployment metadata and authenticated application smoke test remain unverified because this session lacks Vercel connector/CLI authorization. See `DEPLOYMENT.md`.

## Blockers and unverified items

Record anything that stops or limits work here (with the date and what would unblock it). Remove an entry only when it is resolved.

| Date | Blocker | Effect | What would unblock it |
| --- | --- | --- | --- |
| 2026-10-08 | `platform.tripo3d.ai` and Tripo docs are blocked by the egress proxy; only search snippets were readable. | The Tripo adapter was moved to **v3** from secondary sources (create path, `model` name, `output.model_url`, error codes 2010/2000, asset hosts, no cancel, price ≈ $0.30/model) — all **UNVERIFIED**, implemented leniently. | Allow Tripo docs through the proxy, or run one real task and compare the JSON (DEPLOYMENT smoke test). |
| 2026-10-09 | No Tripo API key/account in this environment. | No live hosted call has ever been made. Only mocked documented-contract tests exist; `verified` is hard-wired `false`. | A paid `TRIPO_API_KEY` + `TRIPO_ENABLED=true` + `SIFT_ACCESS_CODE` in a trusted environment, then the manual smoke test in `DEPLOYMENT.md`. |
| 2026-10-09 | All WebGL checks ran in headless Chromium on SwiftShader (software rendering) in this container. | Real-GPU frame rates, memory pressure, mobile GPUs, Safari/Firefox, and the `deviceMemory` heuristic for 1920×1080 are untested. | Manual pass on real desktop + mobile devices/browsers; record results here. |
| 2026-10-09 | `vercel.com` / `api.vercel.com` are unreachable and no Vercel token is available. | Build logs and the deployed app (behind Deployment Protection) cannot be inspected by the agent; deploy health is inferred from GitHub commit statuses only. | An authorized Vercel identity (`vercel inspect <id> --logs`), or the user pasting logs/errors. |

## Working now

- **Contracts and migration.** Canonical v2 schemas (`BuildingSpec`, `Artifact`, `GenerationJob`, `DesignNode`/ports, `DesignRevision`, `SiftProjectV2`), graph connection/cycle validation, lossless v1→v2 migration. Storage writes v2 (`projects-v2`), reads legacy `projects-v1` read-only, preserves unreadable records, saves in one atomic transaction. v1 types and `legacy-fixtures.ts` exist only for migration and its tests.
- **Geometry.** `computeLayout(BuildingSpec)` (rectangle/circle footprints, podium/tower volumes, offsets, twist, taper, setbacks, roof, glazing, clamping warnings) rendered by `three-building.ts`; `deriveBuildingSpec` maps prompts to five typologies. Preview, GLB export and the massing node all use `BuildingSpec`.
- **Studio.** Dashboard (new/open/rename/delete, samples open as editable copies, autosave badge); typed executable canvas (prompt/generation/variation/model/render nodes, validated wiring, Run creates an immutable artifact + job + revision, stale detection, persisted viewport); contextual inspector with validated geometry edits that always create a new artifact/revision; non-destructive branching with labelled lanes and "Use this version"; graph-level undo/redo (ADR-013).
- **Viewer and render.** Focus mode, bounds-based framing, five camera presets (three orthographic), display modes, keyboard controls; Render nodes render offscreen at exact sizes and persist PNGs outside the project record; PNG/GLB export.
- **Persistence.** Everything (graph, viewport, specs, revisions, jobs, settings, render PNGs, hosted GLBs) lives in browser IndexedDB only; JSON backup/import; no cloud sync until P1.01.
- **Hosted generation (Tripo only, ADR-018; P1.02) — unverified against a live account.** Provider picker with configured/unverified/estimated-cost display, fail-closed config (flag + key + `SIFT_ACCESS_CODE`), shared per-IP/daily limits, paid-confirmation naming Tripo, polling, reload-resume (re-enter access code), GLB ingestion, viewing/download. Cancel only stops waiting (Tripo has no known cancel). Projects saved with removed providers open as Local procedural; their leftover active jobs fail with a readable message. Mocked contract tests only. `POST /api/generate` with `provider: "procedural"` answers `{kind:"local"}` (generation runs in the browser; no key/network).
- **Robustness.** Explicit UI and recovery for unsupported WebGL, context loss, render errors, over-complex models, storage failure, missing credentials, provider failure; GPU resources are disposed.
- **Samples.** *Terraced Tower Study* (two branches, self-rendering Render node), Twin Towers, Cylindrical Residence, Spiral Habitat, River Archive.

## Hosted setup tooling

`npm run check:hosted` (`scripts/check-hosted-config.mjs`) is an offline preflight for Tripo configuration (flag, key, access code, `NEXT_PUBLIC_` mistakes, ignored removed-provider variables); it makes no vendor calls. It only checks configuration — E1 stays open until a real-account smoke test is recorded.

## Known limitations

- Tripo has not been called with a real account; status must remain "unverified". Hosted results are fixed meshes (not editable); variation/render nodes need the Local provider's parametric spec. The access code is held in memory only, so after a reload it must be re-entered to resume polling. Rate limits are per server instance (in-memory).
- Courtyard voids are not modelled; vertical/grid facades render as ribbon glazing.
- The viewer's own "PNG" button captures the live canvas at its on-screen size (Render nodes are the resolution-specific path); live viewer settings are not persisted; renders run on the main thread and block briefly at large sizes.
- Orphaned render assets (superseded renders) are kept until the project is deleted; deleted projects are not recoverable; a blank new project is not persisted until its first Generate (schema requires a prompt).
- Procedural output is conceptual massing, not BIM, code-compliant, structural, or fabrication geometry.
- E2E runs only in Chromium on software rendering (SwiftShader); no Firefox/WebKit/mobile or real-GPU coverage; no CI workflow is committed (it would need workflow-scope push permission).
- The Vercel deployment is access-protected and has not been smoke-tested behind protection from this session.
- The legacy Python/PyTorch prototype (`*.py`, `pytorch_gan.ipynb`, `results/`) remains at the repo root pending an owner decision (AGENTS.md forbids silent deletion).

## Verification

Last full run (2026-10-09, fresh clone of the pushed branch after the reference-style redesign): `npm ci` → eslint `--max-warnings=0` clean → `tsc --noEmit` clean → `vitest run` 22 files / 252 tests → production build → 33 Playwright tests → `npm audit` 0 vulnerabilities. All hosted behavior is mocked; Tripo remains unverified live.

## Next action (handoff to the next session)

**Owner decision recorded:** keep E1–E4 outstanding. Nothing is left to build for P0; remaining work is closing exceptions, each of which needs something this environment lacked. Pick up in this order, recording results in the blockers table and a new HANDOFF entry:

1. **E1 — live Tripo:** needs a throwaway low-credit `TRIPO_API_KEY` in a trusted environment (never commit it). Run the smoke test in `DEPLOYMENT.md`. Compare the real JSON against `normalizeTripoTask` in `src/lib/providers/tripo.ts` (`output.model_url`, envelope codes, `Expires` handling are unconfirmed) and adjust the adapter/tests; confirm or correct the ≈ $0.30 cost label. Only then consider changing the hard-wired `verified:false`.
2. **E3 — deployed app:** with an authorized Vercel identity run `vercel inspect <dpl> --logs`, or smoke-test with `E2E_BASE_URL=https://<host> VERCEL_AUTOMATION_BYPASS_SECRET=<secret> npx playwright test e2e/acceptance.spec.ts -g "1\.|8\."`. Also clear the project-level Output Directory override (`public`) in Vercel settings (`vercel.json` currently masks it).
3. **E2 — real devices/browsers:** manual pass on a real GPU, an iPhone/Android, Safari and Firefox (viewer, render PNGs at 1920×1080, touch orbit/pan, autosave, backup/import). Record findings; fix what breaks.
4. **E4 — deviations:** (toolbar done) optionally build the server-side LLM interpreter (see ADR-014) — only if the owner wants them.

Do not start P1.01 (Supabase sync) or P1.03 (GLB import) while any of the above is undecided. **Exception — owner-approved 2026-10-09: P1.02 hosted generation may proceed; scope reduced to Tripo only by ADR-018 (Meshy, Hunyuan3D, HY 3D removed).** It does not close E1–E4. **P1.02 is implemented (mocked only); next for it is a real-account Tripo smoke test (E1).** Latest verified counts are under Verification.

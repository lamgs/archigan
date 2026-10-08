# Handoff log

Append newest entries first. Keep facts in `STATUS.md`; use this log for what changed and what the next agent should do.

## 2026-10-09 — Claude — P0.20 acceptance automation

- Added `@playwright/test`, `@axe-core/playwright`, jsdom + Testing Library (dev deps; `npm audit` clean), `playwright.config.ts` (production build via `next start`, container Chromium auto-detected), `e2e/helpers.ts` (IDB reader, canvas fingerprint, PNG decoder, minimal GLB builder, hosted-API fake, error watcher) and `e2e/acceptance.spec.ts` with one test per PRD scenario; `npm run test:e2e` / `test:all`; evidence screenshots + rendered PNGs in `docs/evidence/`.
- Findings fixed along the way: `--muted` text colour failed WCAG AA (4.27:1) → darkened; “Add next” label on the dark Generation node was ~2:1 → fixed; stale Model-inspector copy updated. Test-side races fixed (dashboard sample vs saved card while loading; reload inside the 900 ms autosave window).
- Checks: lint clean, tsc clean, build passes, 180 unit/component tests, 10/10 e2e (stable over repeated runs, ~47 s).
- Honest limits: scenario 9 is a mocked contract (live Meshy unverified); WebGL on SwiftShader only; no CI workflow committed; GLB scenario compares triangle/mesh counts and bounds with `computeLayout`, not pixel parity with the viewer.
- Next: P0.21 portfolio completion, then the P0 gate review.

## 2026-10-09 — Claude — P0.19 robust states and performance

- Added `limits.ts` (triangle/mesh budgets; enforced in `applyEdit`, viewer, GLB export; hosted GLB triangle cap), `backup.ts` (JSON export/import with validation, id-collision copy, missing-asset warnings), `probeStorage`, `viewer-fallback.tsx` (`ViewerFallback`, `ViewerBoundary`), `app/error.tsx`; viewer now probes WebGL, handles context loss with a restore button, uses `frameloop="demand"`, dpr cap, content-keyed spec (stops needless rebuilds on every autosave), PNG error feedback, GPU diagnostics hook `window.__siftGpu` (counts only). Banner for storage unavailable/save failed with Download backup / Retry save; dashboard “Import backup”.
- Checks: test 163/163, lint clean, build passes. Playwright: no-WebGL browser (fallback, GLB download, inspector, render-panel message), no-IndexedDB browser (banner, backup file), import backup, GPU geometry stable after heavy churn, idle 0 frames/2.5 s, context-loss → restore, 20 sequential 1024² offscreen renders OK.
- Found/fixed: viewer rebuilt the whole model on every unrelated state change; triangle cap (120k) was unreachable under the 240-floor schema cap → set to 60k.
- Open risks / blockers (see STATUS table): software-rasterizer-only GPU testing; backups exclude binary assets; Playwright scripts are still scratch files (P0.20 commits them).
- Next: P0.20 acceptance automation.

## 2026-10-09 — Claude — P0.18 Meshy async lifecycle (unverified live)

- Server: rewrote `providers/meshy.ts` (lenient normalization, error mapping, timeouts, allowlisted size-capped GLB download), added `providers/guard.ts` (fail-closed config, access code, per-IP + daily limiter) and `hosted-http.ts`; routes `POST /api/generate`, `GET|DELETE /api/generate/[taskId]`, `GET .../model`; `/api/providers` now also reports `accessCodeRequired`.
- Client: `hosted.ts` (pure job state machine, timeouts, backoff), `hosted-client.ts`, `paid-confirm.tsx` (explicit confirmation naming Meshy), inspector Hosted section, polling/ingest/cancel/resume in the shell, hosted GLB viewing in `model-preview.tsx` (GLTFLoader) and download. Jobs persist immediately on creation; schema gained optional `createdAt/updatedAt/outputExpiresAt` on jobs.
- Checks: test 153/153 (mocked contract + state tests), lint clean, build passes. Playwright against a **mocked** `/api/generate` verified: dialog before any request, 402 error, queued cancel, immediate job persistence, progress + 429 retry, reload-resume after re-entering the code, ingestion, viewing, reload restore.
- BLOCKERS (also in STATUS): `docs.meshy.ai` blocked by proxy (response field names unconfirmed), no Meshy key (no live call ever made), no Vercel log access. Manual smoke-test checklist added to `DEPLOYMENT.md`.
- Open risks: rate limits are per-instance; hosted meshes are not editable and cannot feed variation/render nodes; access code must be re-entered after reload; SSE streaming not implemented (polling only).
- Next: P0.19 robust states and performance.

## 2026-10-09 — Claude — blockers recorded

- Per the user’s instruction, blockers are now tracked in `docs/STATUS.md` → “Blockers and unverified items” (Meshy docs unreachable, no Meshy key, no Vercel access). Append new ones there and mention them in the handoff entry.

## 2026-10-09 — Claude — P0.17 complete persistence

- Added autosave (debounce + visibilitychange/pagehide flush + flush before navigating away), `projectSignature` change detection, save badge, persisted `settings.viewer` (optional in schema), controlled `ModelPreview` settings. Fixed a real race where an in-flight save’s `setMeta` rolled back newer edits (found by Playwright): saves now merge only name/updatedAt/artifacts/revisions.
- Answered user questions: storage is IndexedDB in each visitor’s browser (local, per browser profile + origin); no Supabase table is needed now; sync is P1.01.
- Checks: test 103/103, lint clean, build passes. Playwright: edit prompt/move node/pan/viewer changes → refresh without pressing Save → everything restored; flush on navigate keeps edits; localStorage/sessionStorage empty.
- Open risks: sample copies are not persisted until first change; autosave writes whole project each time (fine at current size); superseded render assets accumulate; unload-time IndexedDB writes are best-effort.
- Next: P0.18 Meshy lifecycle (paid-request guard first; the unauthenticated `/api/generate` is the main open security risk from the initial review).

## 2026-10-09 — Claude — P0.16 render artifact pipeline

- Added `render-settings.ts` (pure: settings parsing, `supportedResolutions`, `renderInputKey`), `render-image.ts` (offscreen three.js renderer + GPU probe), asset store functions in `storage.ts`, `recordRender` and render freshness (`pending` status) in `workflow.ts`, Render node button and inspector Render panel (settings, preview, download).
- Checks: test 97/97, lint clean, build passes. Playwright (headless Chromium/SwiftShader): decoded PNGs were exactly 1600×900, 1024×1024, 1920×1080; transparent background has real alpha; settings change / upstream edit → “Needs render”; reload restores the image from IndexedDB; download works; no page errors.
- Open risks: 1920×1080 availability relies on `maxRenderbufferSize`/viewport dims/`deviceMemory` heuristics, not an allocation test beyond a canvas-size check; render runs on the main thread; PNG geometry/view match is verified by dimensions/bbox/visual inspection, not an automated pixel diff in CI (P0.20).
- Next: P0.17 persistence audit — write a test that rebuilds the full Terraced Tower board with both branches + a render from storage.

## 2026-10-09 — Claude — P0.15 expanded 3D viewer (+ PR #1 merged, Vercel verified)

- Merged PR #1 (P0.09–P0.14 + Vercel fix) to `main`; Vercel status for the merge commit `47e43aa` was `success`. Branch restarted from `main` for new work.
- Added `viewer.ts` (pure presets/framing/metrics), view modes in `three-building.ts`, rewritten `model-preview.tsx` with a manually managed camera/controls rig (OrbitControls caches `up`, so the camera + controls are rebuilt per preset; frustum/aspect synced on resize).
- Checks: test 85/85, lint clean, build passes. Playwright screenshots confirm top/front/right/axonometric/perspective, all four modes, helpers, focus mode (dialog, Esc, focus returns to trigger), keyboard orbit/zoom/frame, and outer-canvas isolation. Console shows only the pre-existing `/favicon.ico` 404.
- Open risks: shadows are heavy in top view; viewer state is local (not persisted, not shared with the render node yet); GLB export ignores display mode by design; focus mode is a simple overlay without a full focus trap.
- Next: P0.16 render artifacts — reuse `viewer.ts` presets/modes so a render node can reproduce a view at 1024×1024 / 1600×900 offscreen.

## 2026-10-09 — Claude — P0.14 non-destructive branching

- Added `branchFrom`, `commitVariations`, `upstreamArtifactId`, `lineageOf`, `versionsOf`, `restoreVersion` to `workflow.ts`; Branch button on nodes, lane labels, versions fieldset in the inspector; `persist` now snapshots changed variations before saving (also triggered by textarea blur and inspector edits).
- Checks: test 71/71, lint clean, build passes. Playwright: branch → edit branch B floors, follow-up prompt on branch A → distinct previews → save → reload → identical geometry, labels, 6 nodes / 5 edges, correct per-branch version history; no page errors.
- Open risks: variation display still derives from its recipe (snapshots are lineage records, not the render source); no UI to delete a branch other than Backspace on nodes (artifacts remain); snapshots accumulate with each recipe change; blur-triggered autosave also saves other pending edits.
- Next: P0.15 expanded viewer.

## 2026-10-09 — Claude — P0.13 contextual inspector (+ Vercel output directory)

- Added `spec-edit.ts` (typed `SpecEdit`, `applyEdit` validated through `buildingSpecSchema`, replay/merge helpers), `editNodeGeometry` in `workflow.ts`, and `inspector.tsx`; canvas shrinks to make room for the inspector; provider switch moved from the header into the Generation inspector.
- Vercel: user reported `No Output Directory named "public"`; `vercel.json` now sets `outputDirectory: ".next"`. Commit `b0fc5f3` reports GitHub status `success`; app behavior behind protection still unverified. Recommended: also clear the Output Directory override in Project Settings.
- Checks: test 63/63, lint clean, build passes. Playwright: per-node inspector contents (no irrelevant controls), generation edit → revision + persisted, invalid edit message, variation edit leaves source unchanged, collapse/expand, reload restores both.
- Open risks: no UI to browse/restore earlier revisions yet (P0.14); each committed edit on a Generation node adds an artifact (no pruning); inspector volume selector defaults to the first volume.
- Next: P0.14 branching — show lineage and let a design fork into two visible branches.

## 2026-10-09 — Claude — P0.12 typed executable canvas

- UI moved onto `SiftProjectV2`: new `workflow.ts` (evaluate/connect/add/run/preview), typed handles in `studio-node.tsx`, rewritten `studio-shell.tsx` (add toolbar, Add-next menu, `isValidConnection`, viewport persistence). Storage API is v2-native; v1 samples moved to `legacy-fixtures.ts`; samples/projects are v2.
- Checks: test 54/54, lint clean, build passes. Playwright: example → Run → preview, contextual Render add, invalid port drag rejected, prompt edit → out-of-date → re-run, pan + save + reload restores viewport/nodes/edges; no page errors.
- Open risks: prompt dock removed (prompt now lives in the Prompt node); `/api/generate` legacy; render node inert; no per-node delete button (Backspace works).
- Next: user reported failing Vercel preview deployment — debug first; then P0.13/P0.14.

## 2026-10-09 — Claude — P0.11 project dashboard and CRUD

- Added `dashboard.tsx`, `projects.ts` (name validation, unique names, blank/sample-copy/rename helpers), `deleteProject` with tombstones in `storage.ts`, and `fake-indexeddb` (dev dep) storage tests. Shell now opens on the dashboard; brand/“All projects” returns to it; blank projects show example-brief chips and an empty preview.
- Checks: test 43/43, lint clean, build passes. Playwright flow verified first run → example → generate (autosave) → rename (incl. blank-name error) → reload → open → delete with confirmation → reload → sample copy; no page errors.
- Open risks: deletion is permanent; rename uniqueness is checked against loaded list only; `saveProject` of an open project whose record was deleted elsewhere revives it (intended).
- Next: P0.12 typed canvas (prompt/generation/model/render/variation nodes, port validation in `onConnect` using `graph.ts`, persisted viewport) — this is where the UI should move onto `SiftProjectV2`.

## 2026-10-08 — Claude — P0.10 architectural geometry engine

- Added `geometry.ts` (pure layout), `typologies.ts` (`detectTypology`, `deriveBuildingSpec`, `describeSpec`), `three-building.ts` (mesh builder + disposal); switched `model-preview.tsx` and `studio-shell.tsx` to `BuildingSpec`. `three-massing.ts` is retained but unused.
- Checks: test 34/34, lint clean, build passes; Playwright screenshots confirmed four visibly distinct typologies, no page errors.
- Open risks: spec is derived, not persisted; facade vertical/grid and courtyard voids unimplemented; slight shimmer on glazing bands at distance; viewer framing still fixed camera (P0.15).
- Next: P0.11 dashboard/CRUD, then P0.12–P0.14 to move the UI onto v2 and persist specs.

## 2026-10-08 — Claude — P0.09 canonical contracts and migration

- Added v2 Zod contracts to `contracts.ts`, port/connection/cycle validation in `graph.ts`, and v1→v2 migration plus `toLegacyProject` projection and `reconcileStores` in `migrate.ts`.
- Hardened `storage.ts`: atomic `update`, no dropping of unreadable records, legacy key kept read-only (fixes the silent-data-loss risk found in review). UI is unchanged.
- Checks: `npm run test` 20/20, `npm run lint` clean, `npm run build` passes. IndexedDB behavior itself is untested.
- Open risks: legacy edges that break port rules are dropped on migration (UI `onConnect` still allows arbitrary wiring); `BuildingSpec` from migration is approximate; project list capped at 30.
- Next: P0.10 geometry engine on `BuildingSpec`; consider adding `fake-indexeddb` tests. Review items not yet addressed: paid Meshy endpoint guard/status route (P0.18), tracked `__pycache__` files.

## 2026-10-08 — Codex — Vercel deployment recorded

- Recorded the user-created `booth-os/archigan` Vercel project, deployment URL, and dashboard link in `DEPLOYMENT.md` and README.
- Verified the deployment hostname reaches Vercel but redirects unauthenticated requests to Vercel SSO/Deployment Protection.
- Vercel connector inspection returned `403` for the `booth-os` scope; the fallback CLI is not installed. Target environment, source commit, logs, and protected application behavior therefore remain unverified.
- Tests: not rerun because only Markdown documentation changed.
- Next deployment action: use an authorized Vercel identity and the checklist in `DEPLOYMENT.md`; do not disable protection solely for automation.

## 2026-10-08 — Codex — Master brief reconciliation

- Added `PRODUCT_REQUIREMENTS.md` as the durable, implementation-oriented source for the full user-supplied master brief.
- Rebuilt `IMPLEMENTATION_PLAN.md` into Phases 0–6 with delivered baselines, remaining work, and strict exit gates.
- Reprioritized `TASKS.md`: the current app remains a verified foundation, while canonical contracts, richer geometry, project CRUD, typed canvas execution, inspector, branching, expanded viewer, render artifacts, complete persistence, Meshy lifecycle, robust states, acceptance automation, and portfolio completion are explicit P0 work.
- Updated agent rules, decisions, and status so neither Codex nor Claude Code can mistake the existing vertical slice for the master-brief definition of done.
- Tests: not rerun because this handoff changed Markdown coordination files only.
- Next: implement P0.09 canonical contracts/schema migration, followed by P0.10 architectural geometry.

## 2026-10-08 — Codex — Sift 2.0 foundation

- Preserved the clean legacy PyTorch project and documented it as historical/reference-only.
- Added shared agent rules, roadmap, task acceptance criteria, architecture/contracts, decisions, status, and this handoff log.
- Added the first local-first Next.js vertical slice: workflow graph, procedural 3D, refinement, local projects, samples, exports, and server-only provider boundary.
- Verified official public docs describe xFigura as a node-based multi-model workspace and Meshy v2 Text-to-3D as an async preview/refine API. No live Meshy account call was made.
- Checks: `npm run test` passed (6/6), `npm run lint` passed, `npm run build` passed, and `npm audit --audit-level=high` reported zero vulnerabilities.
- Browser smoke test: prompt + refinement regenerated a 20-level glass tower, IndexedDB save updated the local project list, R3F rendered successfully, and GLB serialization reached the completed download state.
- Next: add automated browser end-to-end coverage, or configure a paid Meshy test key and add explicit credit confirmation before attempting the P1 live-provider smoke test.


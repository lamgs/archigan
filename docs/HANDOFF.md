# Handoff log

Append newest entries first. Keep facts in `STATUS.md`; use this log for what changed and what the next agent should do.

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


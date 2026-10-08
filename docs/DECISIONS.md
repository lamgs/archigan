# Architecture decision record

## ADR-001 — Preserve rather than repurpose the legacy GAN

**Status:** accepted, 2026-10-08

The existing repository is a 2018-era PyTorch voxel GAN that requires proprietary training data. It remains as historical reference and is not part of Sift's runtime. Sift does not train on or collect new data.

## ADR-002 — Local-first procedural provider

**Status:** accepted, 2026-10-08

The default experience is a deterministic Three.js massing generator. This makes the primary loop instant, free, testable, and useful offline after installation. Hosted AI augments rather than gates the product.

## ADR-003 — Next.js App Router and server-only provider adapters

**Status:** accepted, 2026-10-08

One TypeScript application supplies UI and secure route handlers. Provider keys stay in server environment variables; the browser calls internal routes only.

## ADR-004 — React Flow is the workflow representation

**Status:** accepted, 2026-10-08

The visible graph mirrors the generation lifecycle and leaves room for architectural constraints without becoming a general-purpose multimodal graph editor.

## ADR-005 — IndexedDB before accounts/backends

**Status:** accepted, 2026-10-08

Projects persist locally through an adapter. This minimizes infrastructure and privacy surface during product discovery. The adapter allows later sync without coupling components to IndexedDB.

## ADR-006 — Meshy is optional and currently unverified

**Status:** accepted, 2026-10-08

Official documentation currently exposes asynchronous Text-to-3D preview/refine endpoints and GLB outputs. The adapter contract is documented and server-scoped, but this repository has not exercised it with a real API key. Product copy must say “configured” and “unverified,” never “connected” or “working,” until a live smoke test passes.

## ADR-007 — Focused architectural product boundary

**Status:** accepted, 2026-10-08

Sift borrows xFigura's spatial workflow clarity, not its broad text/image/video/3D scope. Video and unrelated multimodal features are excluded to keep the brief-to-massing loop coherent.

## ADR-008 — Master brief defines the MVP gate

**Status:** accepted, 2026-10-08

The complete Sift 2.0 master brief is distilled in `PRODUCT_REQUIREMENTS.md` and takes precedence over earlier shorthand plans. The existing local vertical slice is a verified foundation, not the completed MVP. MVP status requires validated architectural specifications, direct parameter edits, non-destructive branching, typed artifact lineage, expanded viewer controls, render artifacts, complete restore, provider lifecycle handling, and the ten acceptance scenarios.

## ADR-010 — v2 storage key and legacy-store handling

**Status:** accepted, 2026-10-08

v2 projects persist under `projects-v2`; the legacy `projects-v1` key is read-only and never deleted, so a rollback loses nothing. Records that fail validation are written back untouched instead of dropped. Legacy edges that violate the new port rules are dropped from the migrated graph with a warning (the original stays in `projects-v1`). `BuildingSpec` and the exact legacy `MassingSpec` are held in the building-spec artifact's `metadata` until P0.10 defines geometry storage.

## ADR-009 — Immutable artifacts and non-destructive revisions

**Status:** accepted, 2026-10-08

Building specifications, GLB models, and PNG renders are immutable artifacts. Edits create child revisions connected to their source. Provider job state is mutable execution metadata and is stored separately from artifact state. This separation prevents silent design loss and makes graph lineage and recovery testable.

## ADR-011 — Bundled samples are built from domain functions; featured sample renders itself

**Status:** accepted, 2026-10-09

`samples.ts` builds the featured *Terraced Tower Study* with the same functions the UI uses (`createWorkflowProject`, `branchFrom`, `editNodeGeometry`, `commitVariations`), so samples can never drift from real behaviour and stay deterministic. Samples contain no binary assets: the Render node is flagged `autoRender` and renders once when the sample is opened, so the image always matches the current geometry code. Opening a sample copies it under a new project id (node/artifact ids are only unique per project and are kept). The earlier *Courtyard Commons* sample was removed because courtyard voids are not modelled (it implied a capability that does not exist); *River Archive* and *Spiral Habitat* remain, and *Twin Towers on a Shared Podium* and *Cylindrical Residence* were added so three typologies are demonstrated. Sample wording uses only words the local interpreter honours (see `INTERPRETER_HELP`).

## ADR-012 — Responsive layout stacks below 900 px

**Status:** accepted, 2026-10-09

Below 900 px the board, inspector, and viewer stack vertically (the page scrolls) instead of overlapping; the left rail hides below 1320 px (samples/projects are on the dashboard); the minimap hides on small screens; the inspector collapses to a slim tab until a node is selected. Responsive CSS lives at the end of `globals.css` so it overrides the base rules.

## ADR-013 — Undo/redo is graph-level; artifacts are never rewound

**Status:** accepted, 2026-10-09 (found missing by the P0 gate review: the plan required "undo/redo or equivalent safe recovery")

Undo/redo snapshots the board (node positions, params, artifact pointers, edges) before each user edit — add/connect/delete, drag, branch, run, geometry edit, version restore, render setting, and one step per typing burst — and restores it on undo. Artifacts, jobs, and revisions are append-only and are *not* rewound, so an undone edit's artifact remains in the version list and the lineage stays truthful. Ctrl/⌘+Z is ignored inside text fields so native field undo keeps working. History is per open project and is cleared on opening another project.

## ADR-014 — The local interpreter reads numbers; no hosted language interpreter

**Status:** accepted, 2026-10-09

The principal sample's brief (12-story, four-story podium, eight-story tower, setbacks every two floors, glazed facade) is now understood by the offline interpreter (`parseCounts`), so the sample is produced by the same path any user brief takes. The optional server-side Anthropic interpreter listed in the implementation plan (Phase 3) was **not built**: it is optional, would add a second paid provider surface, and the app is complete without it. The UI lists exactly which words the local interpreter honours (`INTERPRETER_HELP`) and says other words are ignored. Revisit after the MVP gate if free-form language is wanted.

## ADR-015 — Untrack generated Python bytecode

**Status:** accepted, 2026-10-09

`__pycache__/*.pyc` files (generated bytecode from the legacy prototype, 19 files) were committed by the original repository. They are not source and were removed from version control and git-ignored. Legacy `.py` sources, the notebook, and sample result images are untouched and still preserved as provenance.

## ADR-016 — Hosted generation will be multi-provider: Hunyuan3D, Tripo, Meshy

**Status:** accepted by the owner, 2026-10-09 (implemented 2026-10-09 as P1.02; see ADR-017)

The owner chose hosted AI generation with three providers: **Hunyuan3D (via fal.ai)**, **Tripo**, and **Meshy** (already integrated, still unverified live). Rodin was rejected (API needs a ~$120/month Business plan and targets image-to-3D). Local procedural generation stays the free, default, editable path. Selection basis was price per generation and subscription requirements; approximate figures from public pages (all unverified, re-check before building): Hunyuan3D on fal ≈ $0.225 (Rapid) / $0.375 (Pro) per generation; Tripo ≈ $0.28–0.35 per text-to-3D via API pay-as-you-go (100 credits = $1; the free 300 monthly credits are Studio-only, not API); Meshy ≈ 20 credits for a preview (~$0.40 at Pro rates) and the API requires the Pro plan ($20/month). Self-hosting Hunyuan3D 2.x is free but needs a GPU server and its license excludes the EU, UK and South Korea, so it is out of scope for the Vercel-hosted app. Rules carried over from ADR-006/the Meshy work: server-only keys, fail-closed configuration, shared access code, explicit paid-request confirmation naming the provider, normalized job states, local ingestion of the model, and **nothing is labelled verified without a real-account test**.


## ADR-017 — Provider-neutral hosted interface, extended provider enum, rollback rule

**Status:** accepted, 2026-10-09 (implements ADR-016)

- **Server interface.** `src/lib/providers/types.ts` defines `HostedProvider` (`create`, `status`, `cancel`, plus `label`, `costLabel`, `supportsCancel`, `assetHosts`, `maxGlbBytes`, `config(env)`); `registry.ts` maps ids to adapters. Each adapter owns normalization, error mapping, asset-host allowlist and size cap. Download (host allowlist → size cap → `glTF` magic number, redirects refused) is shared in `http.ts`. Meshy was moved onto the interface without behavior changes (its tests are unchanged), with one tightening: GLB downloads no longer follow redirects, so a signed URL on an allowlisted host cannot bounce the server to another host. `MeshyError` is now an alias of `ProviderError`.
- **Provider ids.** `procedural | meshy | tripo | hunyuan3d-rapid | hunyuan3d-pro`. The two Hunyuan3D tiers are separate ids so a persisted job always knows which fal endpoint its task id belongs to. Per-task routes take `?provider=<id>` (default `meshy`, for clients that predate this change); unknown values are 400.
- **Configuration is fail-closed per provider:** its own flag (`MESHY_ENABLED`, `TRIPO_ENABLED`, `HUNYUAN_ENABLED`) + its own key (`MESHY_API_KEY`, `TRIPO_API_KEY`, `FAL_KEY`) + a shared access code `SIFT_ACCESS_CODE` (fallback `MESHY_ACCESS_CODE`). Per-IP and daily limits (`SIFT_DAILY_LIMIT`, fallback `MESHY_DAILY_LIMIT`) are counted across all providers by one limiter.
- **Backward compatibility and rollback.** Adding values to `providerSchema` is additive: projects saved with `procedural` or `meshy` open unchanged. The reverse does not hold: **a build from before this change cannot parse a project whose `settings.provider` or job `provider` is `tripo`/`hunyuan3d-*`**, so after rolling back a deployment such projects fail validation and open as unreadable (they are preserved in IndexedDB, not deleted). Before rolling back, switch affected projects back to Local/Meshy or accept that they are unreadable until roll-forward. No schema version bump was made because old data remains valid.
- **Verification status.** fal.ai and Tripo's web documentation remains unreachable (egress blocked). Tencent's official international Go SDK confirms part of Hunyuan3D Pro's native upstream schema, but it does not define fal.ai's wrapper routes, response envelope, Rapid tier, cancellation, or billing. Tripo's newer official JS/TS SDK v0.3.0 explicitly targets v3 rather than the older `/v2/openapi/task` API, so the adapter uses `https://openapi.tripo3d.ai/v3`, `POST /generation/text-to-model`, `GET /tasks/{id}`, and primary `output.model_url`; legacy output names remain accepted during migration. The SDK confirms code 2010 as insufficient credits but does not settle prompt limits, asset hosts, price, other envelope-code meanings, or live behavior. All adapters remain labelled UNVERIFIED until real-account smoke tests; documentation and mocks alone never change that status.

## ADR-018 — Use Next's TypeScript compiler-API checker while the project is on TypeScript 6

**Status:** accepted, 2026-10-08

Next 16.4 defaults to spawning the project-local `tsc` CLI and parsing `tsc --showConfig`. In the managed verification environment, stdout from Node child processes is unavailable, so the CLI exits successfully with an empty captured result and every `next build` stops at “Could not parse output from TypeScript's --showConfig.” The project remains on TypeScript 6, which exposes the compiler API, so `next.config.ts` sets the documented `experimental.useTypeScriptCli: false`. Production builds still perform full type checking; this is not `ignoreBuildErrors`. Revisit this decision before adopting TypeScript 7, whose JavaScript compiler API is currently unavailable.

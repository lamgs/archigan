# Deployment

## Vercel environment

- **Project:** `archigan`
- **Team:** `booth-os`
- **Deployment URL:** <https://archigan-ctjx7uh6t-booth-os.vercel.app>
- **Dashboard:** <https://vercel.com/booth-os/archigan/HK1XNPR5cJ75CM18jo433zP48hzX>
- **Recorded:** 2026-10-08

## Current verification status

The deployment hostname is reachable on Vercel, but an unauthenticated request returns `302 Found` to Vercel's SSO endpoint. Treat the environment as **deployed and access-protected**, not publicly verified.

The Codex Vercel connector identified the `booth-os` team but returned `403 forbidden` for project/deployment inspection. The local Vercel CLI is not installed, so this session could not verify:

- whether the deployment is Preview or Production;
- which Git commit it contains;
- build duration and build logs;
- runtime errors after authenticating through Deployment Protection;
- whether a stable production alias exists in addition to the deployment-specific hostname.

Do not disable Deployment Protection merely to make automated verification pass. Use an authorized Vercel connection or `vercel curl` with the existing team identity.

## Build failure investigation (2026-10-09)

GitHub commit statuses (context `Vercel`) show every deployment since `f43fc44` (the first commit with `package.json`) as `failure`; the earlier README-only commit `7fd7a25` succeeded as a static deploy. `npm ci`, `npm run build`, tests, and lint pass locally on Node 22, and the failure predates the P0.09+ changes. Vercel build logs were not reachable from the agent environment (vercel.com/api.vercel.com blocked, no token), so the root cause was initially inferred: the project's Framework Preset/Output Directory most likely still reflected the earlier static project.

**Outcome:** commit `6fa7198` (with `vercel.json`) reports GitHub status `success` ("Deployment has completed"), which supports that diagnosis. Application behavior behind Deployment Protection is still unverified. Earlier failed commits stay failed unless redeployed.

**Update:** the user reported the Vercel error `No Output Directory named "public" found after the Build completed` — confirming the project-level Output Directory is `public`. `outputDirectory: null` did not clear it, so `vercel.json` now sets `.next` explicitly. The proper long-term fix is also to clear the Output Directory override in Project Settings → Build & Development.

Mitigation: `vercel.json` pins `framework: nextjs`, `npm ci`, `npm run build`, and sets `outputDirectory` to `.next`. If the next deployment still fails, run `npx vercel inspect <dpl_id> --logs` (id is in the failing commit status `target_url`) and append the error here. Also check Project Settings → General: Framework Preset = Next.js, Root Directory empty, Node.js version 20.x–24.x.

## Hosted generation (Hunyuan3D, Tripo, Meshy) — configuration and manual live smoke test

Each hosted provider is **disabled unless its own flag and key AND the shared access code are set** (fail closed). Setting one provider never enables another.

```text
SIFT_ACCESS_CODE=<shared secret users type before any paid request; anyone with it can spend credits on every enabled provider>
SIFT_DAILY_LIMIT=20        # optional, per server instance, counted across ALL providers
MESHY_ENABLED=true         MESHY_API_KEY=<server-only secret>
TRIPO_ENABLED=true         TRIPO_API_KEY=<server-only secret>
HUNYUAN_ENABLED=true       FAL_KEY=<server-only secret>     # enables Hunyuan3D Rapid and Pro
```

`MESHY_ACCESS_CODE` / `MESHY_DAILY_LIMIT` remain accepted as fallbacks. Routes: `POST /api/generate` (body `provider`, `confirmSpend: true`; header `x-sift-access-code`), `GET|DELETE /api/generate/{taskId}?provider=<id>` (status / cancel), `GET /api/generate/{taskId}/model?provider=<id>` (GLB ingest; only HTTPS hosts on the provider's allowlist are fetched, redirects refused, 100 MB cap, `glTF` magic check), `GET /api/providers` (secret-free catalog). Provider ids: `meshy`, `tripo`, `hunyuan3d-rapid`, `hunyuan3d-pro`.

**Status: ALL UNVERIFIED.** Only mocked documented-contract tests exist; vendor docs for fal.ai and Tripo (and Meshy) were unreachable when the adapters were written. For each provider, use a throwaway low-credit key in a trusted environment and check off each item, then update `STATUS.md` (remove the blocker) and flip `verified` only if all pass:

1. Create task: `POST /api/generate` returns 202 with a task id; the vendor dashboard shows the task.
2. Status: `GET /api/generate/{id}?provider=…` transitions queued → running → completed; compare the raw vendor JSON with the adapter's normalizer. **Unconfirmed names:** Meshy `model_urls.glb`, `task_error.message`, `expires_at`; Tripo base URL/path (v2 `/task` vs v3 per-capability endpoints), body fields, `output.pbr_model|model|model_url`, envelope error codes; fal endpoint ids, the app-id form of status/result/cancel URLs, `model_glb` / `model_urls.glb`, Rapid `enable_pbr`/`enable_geometry`, Pro `face_count`, prompt limits, the 403 balance wording, and the signed-asset hosts (`fal.media`, `tripo3d.com|ai`).
3. Ingest: `/model` returns a GLB that opens in the viewer and downloads (fal Rapid may return OBJ — the app rejects non-GLB).
4. Cancel: Meshy DELETE on a queued task succeeds (409 when running); fal cancel via `PUT …/cancel`; Tripo has no known cancel (the app only stops waiting).
5. Failure paths: bad key → `auth`; empty credits → `insufficient-credits`; rapid requests → 429 handling.
6. Reload mid-task: the job resumes after re-entering the access code.
7. Record the real per-generation price; correct the approximate cost labels in `src/lib/providers/*.ts` and `src/lib/provider-meta.ts` (Tripo's is currently "not confirmed").

## Release, rollback, and cost runbook

**Release (every merge to `main`)**
1. Locally or in a clean clone: `npm ci && npm run lint && npm run test && npm run test:e2e` (all green).
2. Merge. Vercel builds automatically; confirm the `Vercel` commit status is `success` (GitHub → commit → status) and open the deployment URL while signed in to Vercel.
3. Optional smoke test of the deployed app: `E2E_BASE_URL=https://<deployment-host> VERCEL_AUTOMATION_BYPASS_SECRET=<secret from Project → Deployment Protection> npx playwright test e2e/acceptance.spec.ts -g "1\.|8\."` (scenarios that need only the deployed app; do not point the mocked-hosted scenario at production).

**Rollback.** Vercel → Project → Deployments → pick the last good deployment → *Promote to Production* (or `vercel rollback`), or revert the commit on `main`. User data is not at risk: projects live in each visitor's browser, not on the server. Compatibility rule that makes rollback safe: stored data is only ever *read forward* — never remove read support for schema v1/v2; a future v3 must migrate on read and must not rewrite records in a way an older build cannot open.

**Provider cost control (all hosted providers).** A paid request needs the server-side `SIFT_ACCESS_CODE`, the provider's own flag and key, an explicit user confirmation naming the selected provider, and passes per-IP (3 per 10 min) and daily (`SIFT_DAILY_LIMIT`, default 20) limits that are shared across providers. The limiter is per server instance, so worst-case daily spend is *limit × warm instances × the most expensive provider's price* — also set a spend cap/alerts in each vendor account (Meshy, Tripo, fal.ai). Approximate costs shown in the UI are estimates from public pages (Hunyuan3D Rapid ≈ $0.225, Pro ≈ $0.375, Meshy ≈ 20 credits; Tripo unconfirmed) and must be re-checked. To stop **one** provider immediately: set its flag to `false` (or delete its key) and redeploy; calls then fail closed with 503. To stop **all** hosted spending: unset `SIFT_ACCESS_CODE` (and `MESHY_ACCESS_CODE`). To revoke a leaked access code: change it and redeploy. If a key may have leaked: rotate it at the vendor first, then update the env var. Review each vendor's usage dashboard after any public sharing of the code. **Rollback caveat:** projects saved with `tripo` or `hunyuan3d-*` cannot be opened by builds from before ADR-017 (see `DECISIONS.md`).

**Performance budget.** Building meshes are capped at 60 000 triangles / 1 200 draw calls (`src/lib/limits.ts`); hosted GLB previews at 1.5 M triangles; the viewer renders on demand (0 idle frames) with device-pixel-ratio capped at 1.75; offscreen renders are sequential and release their GPU context; autosave is debounced (900 ms). Regressions in these show up as e2e failures (idle-frame and GPU-resource checks are in `docs/STATUS.md` evidence).

## Runtime configuration

The procedural application requires no secrets. Optional hosted generation requires server-side variables:

```text
SIFT_ACCESS_CODE, plus per provider: MESHY_ENABLED + MESHY_API_KEY, TRIPO_ENABLED + TRIPO_API_KEY, HUNYUAN_ENABLED + FAL_KEY
(configured in Vercel, never committed)
```

Until a real-account test succeeds for a provider, the application and status docs must continue to label it as unverified.

## Authorized verification checklist

From a Vercel-authenticated environment with access to `booth-os`:

```bash
vercel --scope booth-os project inspect archigan
vercel --scope booth-os inspect archigan-ctjx7uh6t-booth-os.vercel.app
vercel curl https://archigan-ctjx7uh6t-booth-os.vercel.app/
```

Then verify:

1. Deployment is `READY` and linked to the intended `main` commit.
2. Home page renders the canvas and procedural model after protection is satisfied.
3. `/api/providers` returns procedural configured/verified and an honest, unverified state for each hosted provider.
4. Prompt generation, IndexedDB save/reload, PNG, and GLB work on the deployed origin.
5. Runtime error logs are clean for the smoke-test window.
6. `STATUS.md` is updated with target, commit, URL, and observed result.


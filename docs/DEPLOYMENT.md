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

## Hosted generation (Meshy) — configuration and manual live smoke test

Hosted generation is **disabled unless all of these server variables are set** (it fails closed):

```text
MESHY_ENABLED=true
MESHY_API_KEY=<server-only secret>
MESHY_ACCESS_CODE=<shared secret users type before any paid request; anyone with it can spend credits>
MESHY_DAILY_LIMIT=20   # optional, per server instance
```

Routes: `POST /api/generate` (needs `x-sift-access-code` + `confirmSpend: true`), `GET|DELETE /api/generate/{taskId}` (status / cancel), `GET /api/generate/{taskId}/model` (GLB ingest; only HTTPS `*.meshy.ai` URLs are fetched). Cancel works only for queued tasks; Meshy answers 409 for running ones and the UI says so.

**Status: UNVERIFIED.** Only mocked documented-contract tests exist. To record a real result, use a throwaway low-credit key in a trusted environment and check off each item, then update `STATUS.md` (remove the blocker) and flip `verified` only if all pass:

1. Create task: `POST /api/generate` returns 202 with a task id; Meshy dashboard shows the task.
2. Status: `GET /api/generate/{id}` transitions queued → running (progress rises) → completed; compare the raw Meshy JSON with `normalizeTask` (`model_urls.glb`, `task_error.message`, `expires_at` field names are **unconfirmed**).
3. Ingest: `/model` returns a GLB that opens in the viewer and downloads.
4. Cancel: DELETE on a queued task succeeds; on a running task returns 409 (`running`).
5. Failure paths: bad key → `auth`; empty credits → `insufficient-credits`; rapid requests → 429 handling.
6. Reload mid-task: the job resumes after re-entering the access code.

## Release, rollback, and cost runbook

**Release (every merge to `main`)**
1. Locally or in a clean clone: `npm ci && npm run lint && npm run test && npm run test:e2e` (all green).
2. Merge. Vercel builds automatically; confirm the `Vercel` commit status is `success` (GitHub → commit → status) and open the deployment URL while signed in to Vercel.
3. Optional smoke test of the deployed app: `E2E_BASE_URL=https://<deployment-host> VERCEL_AUTOMATION_BYPASS_SECRET=<secret from Project → Deployment Protection> npx playwright test e2e/acceptance.spec.ts -g "1\.|8\."` (scenarios that need only the deployed app; do not point the mocked-hosted scenario at production).

**Rollback.** Vercel → Project → Deployments → pick the last good deployment → *Promote to Production* (or `vercel rollback`), or revert the commit on `main`. User data is not at risk: projects live in each visitor's browser, not on the server. Compatibility rule that makes rollback safe: stored data is only ever *read forward* — never remove read support for schema v1/v2; a future v3 must migrate on read and must not rewrite records in a way an older build cannot open.

**Provider cost control (Meshy).** A paid request needs the server-side `MESHY_ACCESS_CODE`, an explicit user confirmation, and passes per-IP (3 per 10 min) and daily (`MESHY_DAILY_LIMIT`, default 20) limits. The limiter is per server instance, so worst-case daily spend is *limit × warm instances* — also set a spend cap/alerts in the Meshy account itself. To stop spending immediately: set `MESHY_ENABLED=false` (or delete `MESHY_API_KEY`) and redeploy; hosted calls then fail closed with 503. To revoke a leaked access code: change `MESHY_ACCESS_CODE` and redeploy. If the API key may have leaked: rotate it in Meshy first, then update the env var. Review Meshy's usage dashboard after any public sharing of the code.

**Performance budget.** Building meshes are capped at 60 000 triangles / 1 200 draw calls (`src/lib/limits.ts`); hosted GLB previews at 1.5 M triangles; the viewer renders on demand (0 idle frames) with device-pixel-ratio capped at 1.75; offscreen renders are sequential and release their GPU context; autosave is debounced (900 ms). Regressions in these show up as e2e failures (idle-frame and GPU-resource checks are in `docs/STATUS.md` evidence).

## Runtime configuration

The procedural application requires no secrets. Optional hosted generation requires server-side variables:

```text
MESHY_ENABLED=true
MESHY_API_KEY=<configured in Vercel, never committed>
```

Until a real Meshy account test succeeds, the application and status docs must continue to label Meshy as unverified.

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
3. `/api/providers` returns procedural configured/verified and an honest Meshy state.
4. Prompt generation, IndexedDB save/reload, PNG, and GLB work on the deployed origin.
5. Runtime error logs are clean for the smoke-test window.
6. `STATUS.md` is updated with target, commit, URL, and observed result.


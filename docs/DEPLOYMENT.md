# Deployment

## Vercel environment

- **Project:** `archigan`
- **Team:** `gabelam` (owner-confirmed 2026-10-08; earlier docs said `booth-os`, which is now outdated)
- **Production domain:** <https://archigan.vercel.app> (stable alias, owner-confirmed 2026-10-08). Per-deployment URLs such as `archigan-<hash>-….vercel.app` change on every build; do not record them here. Take a specific deployment's URL from the dashboard or the commit status `target_url` only when inspecting that build.
- **Dashboard:** <https://vercel.com/gabelam/archigan>
- **Recorded:** 2026-10-08

## Current verification status

The deployment hostname is reachable on Vercel, but an unauthenticated request returns `302 Found` to Vercel's SSO endpoint. Treat the environment as **deployed and access-protected**, not publicly verified.

The Codex Vercel connector identified the old `booth-os` team but returned `403 forbidden` for project/deployment inspection. The local Vercel CLI is not installed, so this session could not verify:

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

## Hosted generation (Tripo) — configuration and manual live smoke test

Hosted generation uses one provider, Tripo (ADR-018; Meshy, Hunyuan3D and HY 3D were removed). It is **disabled unless the Tripo flag and key AND the shared access code are set** (fail closed).

```text
SIFT_ACCESS_CODE=<shared secret users type before any paid request; anyone with it can spend your Tripo credits>
SIFT_DAILY_LIMIT=20        # optional, per server instance
TRIPO_ENABLED=true         TRIPO_API_KEY=<server-only secret>
```

The old `MESHY_*`, `HUNYUAN_ENABLED`, `FAL_KEY` and `TENCENT_*` variables are no longer read; delete them from Vercel. Routes: `POST /api/generate` (body `provider`, `confirmSpend: true`; header `x-sift-access-code`; a removed provider id answers 400 `unsupported-provider`), `GET|DELETE /api/generate/{taskId}?provider=tripo` (status; cancel answers 501 because Tripo documents none; unknown provider 400 `unknown-provider`), `GET /api/generate/{taskId}/model?provider=tripo` (GLB ingest; only HTTPS hosts on the allowlist are fetched, redirects refused, 100 MB cap, `glTF` magic check), `GET /api/providers` (secret-free catalog: `procedural` and `tripo`).

### Setting up (no vendor calls)

1. Create a **throwaway, low-credit** key at Tripo (`TRIPO_API_KEY`) and set a spend cap/alert in the Tripo dashboard. Never paste keys into chat, issues, or the repo.
2. Generate a long random access code (e.g. `openssl rand -base64 24`) for `SIFT_ACCESS_CODE`.
3. Set variables in Vercel → Project → Settings → Environment Variables (Production and/or Preview; mark Sensitive) or, locally, in `.env.local`: `TRIPO_ENABLED=true`, `TRIPO_API_KEY`, `SIFT_ACCESS_CODE`. Redeploy after changing Vercel variables.
4. Run the offline preflight (reads env only; no network to any vendor; never prints secret values): `npm run check:hosted` (add `-- --url https://<deployment>` to also read your own app's `/api/providers`). Tripo should read READY.
5. In the app, the picker should now show Tripo as configured, still labelled **unverified**. Selecting it and confirming the dialog is what first spends money — that is the smoke test below, not setup.

**Status: UNVERIFIED.** Only mocked documented-contract tests exist; Tripo's docs were unreachable when the adapter was written. Use a throwaway low-credit key in a trusted environment and check off each item, then update `STATUS.md` (remove the blocker) and flip `verified` only if all pass:

1. Create task: `POST /api/generate` returns 202 with a task id; the Tripo dashboard shows the task.
2. Status: `GET /api/generate/{id}?provider=tripo` transitions queued → running → completed; compare the raw vendor JSON with the adapter's normalizer. **Unconfirmed names:** Tripo v3 base URL/path (`openapi.tripo3d.ai/v3`, `POST /generation/text-to-model` with `model`, `GET /tasks/{id}` vs v2 `/task`), body fields, `output.model_url` (v2 `pbr_model|model` tolerated), envelope error codes (2010 credits, 2000 rate limit), asset hosts (`tripo3d.com|ai`); results expire ≈5 min after success; no cancel.
3. Ingest: `/model` returns a GLB that opens in the viewer and downloads (the app rejects non-GLB).
4. Cancel: Tripo has no known cancel (the app only stops waiting; DELETE answers 501).
5. Failure paths: bad key → `auth`; empty credits → `insufficient-credits`; rapid requests → 429 handling.
6. Reload mid-task: the job resumes after re-entering the access code.
7. Record the real per-generation price; correct the cost label in `src/lib/providers/tripo.ts` (`TRIPO_COST_LABEL`) and `src/lib/provider-meta.ts` (currently "≈ $0.30 per model (estimate)", unconfirmed, from ADR-016's ≈ $0.28–0.35 at 100 credits = $1).

## Release, rollback, and cost runbook

**Release (every merge to `main`)**
1. Locally or in a clean clone: `npm ci && npm run lint && npm run test && npm run test:e2e` (all green).
2. Merge. Vercel builds automatically; confirm the `Vercel` commit status is `success` (GitHub → commit → status) and open the deployment URL while signed in to Vercel.
3. Optional smoke test of the deployed app: `E2E_BASE_URL=https://<deployment-host> VERCEL_AUTOMATION_BYPASS_SECRET=<secret from Project → Deployment Protection> npx playwright test e2e/acceptance.spec.ts -g "1\.|8\."` (scenarios that need only the deployed app; do not point the mocked-hosted scenario at production).

**Rollback.** Vercel → Project → Deployments → pick the last good deployment → *Promote to Production* (or `vercel rollback`), or revert the commit on `main`. User data is not at risk: projects live in each visitor's browser, not on the server. Compatibility rule that makes rollback safe: stored data is only ever *read forward* — never remove read support for schema v1/v2; a future v3 must migrate on read and must not rewrite records in a way an older build cannot open.

**Provider cost control (Tripo).** A paid request needs the server-side `SIFT_ACCESS_CODE`, Tripo's flag and key, an explicit user confirmation naming Tripo, and passes per-IP (3 per 10 min) and daily (`SIFT_DAILY_LIMIT`, default 20) limits. The limiter is per server instance, so worst-case daily spend is *limit × warm instances × the per-model price* — also set a spend cap/alert in the Tripo account. The cost shown in the UI (≈ $0.30 per model) is an unconfirmed estimate from public pages and must be re-checked. To stop hosted spending immediately: set `TRIPO_ENABLED=false` (or delete the key, or unset `SIFT_ACCESS_CODE`) and redeploy; calls then fail closed with 503. To revoke a leaked access code: change it and redeploy. If the key may have leaked: rotate it at Tripo first, then update the env var. Review Tripo's usage dashboard after any public sharing of the code. **Rollback caveat (ADR-018):** builds from before ADR-017 cannot open projects saved with `tripo`; builds from ADR-017 through before ADR-018 open everything, since legacy provider values stay readable.

**Performance budget.** Building meshes are capped at 60 000 triangles / 1 200 draw calls (`src/lib/limits.ts`); hosted GLB previews at 1.5 M triangles; the viewer renders on demand (0 idle frames) with device-pixel-ratio capped at 1.75; offscreen renders are sequential and release their GPU context; autosave is debounced (900 ms). Regressions in these show up as e2e failures (idle-frame and GPU-resource checks are in `docs/STATUS.md` evidence).

## Runtime configuration

The procedural application requires no secrets. Optional hosted generation requires server-side variables:

```text
SIFT_ACCESS_CODE, TRIPO_ENABLED, TRIPO_API_KEY
(configured in Vercel, never committed)
```

Until a real-account test succeeds for Tripo, the application and status docs must continue to label it as unverified.

## Authorized verification checklist

From a Vercel-authenticated environment with access to `gabelam`:

```bash
vercel --scope gabelam project inspect archigan
vercel --scope gabelam inspect archigan.vercel.app
vercel curl https://archigan.vercel.app/
```

Then verify:

1. Deployment is `READY` and linked to the intended `main` commit.
2. Home page renders the canvas and procedural model after protection is satisfied.
3. `/api/providers` returns procedural configured/verified and an honest, unverified state for Tripo.
4. Prompt generation, IndexedDB save/reload, PNG, and GLB work on the deployed origin.
5. Runtime error logs are clean for the smoke-test window.
6. `STATUS.md` is updated with target, commit, URL, and observed result.


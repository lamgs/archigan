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


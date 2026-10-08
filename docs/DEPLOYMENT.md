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


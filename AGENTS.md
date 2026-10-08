# Sift 2.0 agent guide

This repository is the shared source of truth for Codex, Claude Code, and human contributors.

## Start here

1. Read `docs/PRODUCT_REQUIREMENTS.md`, `docs/STATUS.md`, then `docs/TASKS.md`.
2. Read `docs/IMPLEMENTATION_PLAN.md`, `docs/ARCHITECTURE.md`, and `docs/DECISIONS.md` before changing contracts, sequencing, or dependencies.
3. Check `git status` and preserve unrelated work.
4. Work on the highest-priority unblocked task unless the user names another target.
5. Before stopping, run the relevant checks and update `docs/STATUS.md` plus `docs/HANDOFF.md`.

## Product boundary

Sift 2.0 is a focused architectural prompt-to-3D workflow. It is not a general multimodal studio.

- In scope: project CRUD, text prompts, validated parametric architecture, non-destructive branching, typed React Flow artifacts, expanded 3D viewing, material/camera/render controls, optional hosted 3D generation, PNG/GLB export, complete local restore, sample projects, and polished architectural UI.
- Out of scope: video, model training, collecting new training data, social/community features, and exposing provider secrets to the browser.
- The legacy PyTorch 3D-GAN files at the repository root are historical reference only. Do not wire them into the product or require proprietary training data.

## Engineering rules

- Use TypeScript and the Next.js App Router.
- Keep provider credentials and calls server-side. Never prefix secrets with `NEXT_PUBLIC_`.
- The procedural provider must remain usable without an account or network access.
- Treat the workflow schema in `src/lib/contracts.ts` as a compatibility boundary. Migrate persisted data when it changes.
- Treat artifacts as immutable outputs and create child revisions for edits; never silently replace a source design.
- Keep job execution state separate from immutable artifact state, and validate graph ports/connections in the domain layer.
- Persist user projects locally through the storage adapter; UI components must not call IndexedDB directly.
- Keep provider-specific payloads behind `src/lib/providers/` and API routes.
- Prefer small, testable domain functions over logic embedded in React components.
- Do not claim hosted-provider behavior is verified unless it has been exercised with a real account in the current environment.
- Do not silently delete or rewrite legacy files. Document deliberate migrations in `docs/DECISIONS.md`.

## Required checks

For application changes, run:

```bash
npm run test
npm run lint
npm run build
```

If a check cannot run, record the exact reason in `docs/STATUS.md` and the newest handoff entry.

## Handoff protocol

- Keep `docs/STATUS.md` concise and current; replace stale facts rather than appending a diary.
- Append one dated entry to `docs/HANDOFF.md` with changes, checks, open risks, and the next concrete action.
- Update task checkboxes only when their acceptance criteria are actually met.
- Never call the MVP complete while any P0 task or acceptance scenario in `docs/PRODUCT_REQUIREMENTS.md` remains open.
- Do not commit unless the user explicitly asks. Always report the current branch and dirty/clean state.

<!-- BEGIN:nextjs-agent-rules -->

## This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

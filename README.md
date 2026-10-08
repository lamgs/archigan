# Sift 2.0

Sift is a focused, xFigura-inspired architectural prompt-to-3D workspace. Describe a building, run it into real parametric 3D massing, branch and edit variations on a node canvas, render PNGs, and export a GLB — locally in your browser, with no account. Optional hosted generation (Meshy) is available behind a server-side key.

> **Scope:** conceptual massing for early design. It is not BIM, structural, code-compliance, or fabrication geometry.

## What you can do

- **Prompt → model.** A Prompt node feeds a Generation node; *Run* creates an immutable building specification and a 3D model (terraced tower, twin towers, cylindrical, rotated, or low-rise typologies).
- **Edit non-destructively.** The inspector changes footprint, floors, setbacks, taper, twist, facade, roof, and materials. Edits create new revisions; earlier versions stay available.
- **Branch.** Fork a design into labelled variations (Branch A, Branch B…) that stay visible with their lineage.
- **View.** Orbit/pan/zoom, camera presets (perspective, axonometric, top, front, right), shaded/clay/glass-concrete/wireframe, focus mode, keyboard controls.
- **Render and export.** Render nodes produce 1024×1024, 1600×900, or (when the device supports it) 1920×1080 PNGs; the viewer exports a GLB.
- **Keep everything.** Projects autosave to your browser (IndexedDB) and restore completely, including render images; JSON backup/import is available.
- **Start from the featured sample.** *Terraced Tower Study* opens as a complete board with two branches and a render.

## Quick start

```bash
npm install
npm run dev          # http://localhost:3000
```

The procedural provider needs no environment variables, no account, and no network access. Requires Node 20.9+.

## Architecture

```text
Browser (Next.js App Router client)
  React Flow board ──┐         src/lib/workflow.ts     pure domain: evaluate / connect / run / branch / render
  Inspector ─────────┤────────▶ src/lib/geometry.ts     BuildingSpec → per-floor slabs (no rendering deps)
  3D viewer (R3F) ───┘          src/lib/typologies.ts    prompt → BuildingSpec (local interpreter)
        │                       src/lib/contracts.ts     Zod schemas = compatibility boundary (schema v2)
        ├── src/lib/storage.ts ─▶ IndexedDB (projects + binary assets); the only persistence dependency
        └── /api/generate/*   ─▶ src/lib/providers/*  server-only provider adapters (Hunyuan3D, Tripo, Meshy) + spend guard
```

- **Immutable artifacts.** Building specs, renders, and hosted GLBs are artifacts; edits create child artifacts linked by revisions. Jobs (execution state) are kept separate.
- **Typed graph.** Ports have kinds (`prompt`, `building-spec`, `model-glb`, `render-png`); connections are validated (type, duplicate input, cycles).
- **Local-first.** All project data lives in the visitor's own browser (per browser profile and site address). There is no cloud sync yet; see *Limitations*.
- Deeper detail: [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md), decisions in [`docs/DECISIONS.md`](./docs/DECISIONS.md).

## Credentials (optional hosted generation)

Hosted generation is **off by default** and fails closed, per provider. A provider is enabled only if its own flag **and** key are set **and** a shared access code is set (copy `.env.example` to `.env.local`):

| Provider | Flag | Key |
| --- | --- | --- |
| Hunyuan3D via fal.ai (Rapid and Pro) | `HUNYUAN_ENABLED=true` | `FAL_KEY` |
| Tripo | `TRIPO_ENABLED=true` | `TRIPO_API_KEY` |
| Meshy | `MESHY_ENABLED=true` | `MESHY_API_KEY` |

| Shared variable | Purpose |
| --- | --- |
| `SIFT_ACCESS_CODE` | Secret users must type before any paid request (`MESHY_ACCESS_CODE` still works as a fallback). Anyone who has it can spend credits on every enabled provider |
| `SIFT_DAILY_LIMIT` | Optional per-instance daily cap, counted across all providers (default 20; `MESHY_DAILY_LIMIT` fallback) |

Keys are server-only (never use a `NEXT_PUBLIC_` prefix). Users pick a provider in the Generation inspector and confirm each paid request in a dialog that names the selected provider; costs shown are approximate estimates. **All three hosted integrations are unverified:** they were written from vendor documentation summaries (the fal.ai and Tripo docs were unreachable) and exercised only against mocks, never a live account — see [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) for the smoke test. Hosted results are fixed meshes — viewable and downloadable, not editable. Local procedural generation needs none of this.

## Tests

```bash
npm run lint
npm run test        # unit + component tests (Vitest, jsdom)
npm run build
npm run test:e2e    # builds, then Playwright against the production build (:3200)
```

The Playwright suite covers the ten acceptance scenarios in [`docs/PRODUCT_REQUIREMENTS.md`](./docs/PRODUCT_REQUIREMENTS.md), the featured sample, responsive layouts at four widths, and axe WCAG A/AA scans, and writes evidence screenshots to [`docs/evidence/`](./docs/evidence). In containers with a pre-installed Chromium it is used automatically (`/opt/pw-browsers/chromium`, or set `PLAYWRIGHT_CHROMIUM_EXECUTABLE`); elsewhere run `npx playwright install chromium` first. Hosted-provider tests use a **mock** and never call Meshy.

## Deployment

Sift deploys to Vercel as a standard Next.js app (`vercel.json` pins the framework and output directory). The Vercel project is `gabelam/archigan`; the deployment is access-protected, so see [`docs/DEPLOYMENT.md`](./docs/DEPLOYMENT.md) for the verification checklist and the Meshy variables. Do not commit secrets.

## Limitations

- Conceptual massing only; the local interpreter understands a fixed vocabulary of keywords (listed in the app) and ignores other words.
- Storage is per browser: clearing site data, private browsing, or switching device loses projects unless you downloaded a backup (backups exclude render images and hosted models).
- Hosted generation is unverified live, produces non-editable meshes, and cannot cancel running hosted tasks (Tripo has no known cancel).
- Tested in Chromium on software rendering only; other browsers, mobile GPUs, and the 1920×1080 availability heuristic are not independently verified.
- Courtyard voids and vertical/grid facade fin detail are not modelled.

## Independence and provenance

Sift is an independent project. It is **inspired by** the spatial node-workflow idea of tools like xFigura but is not affiliated with, endorsed by, or built from xFigura or Meshy, and it uses no proprietary data. The repository began as a PyTorch voxel 3D-GAN research prototype ("ArchiGAN"); those Python files are preserved below for provenance only and are not used by, or required for, the product. No models are trained and no training data is collected.

Project coordination for contributors and AI agents lives in [`AGENTS.md`](./AGENTS.md), with current state in [`docs/STATUS.md`](./docs/STATUS.md) and acceptance criteria in [`docs/TASKS.md`](./docs/TASKS.md).

## Legacy research prototype

The original repository was a PyTorch implementation of an early voxel 3D-GAN. Its Python files, notebook, and sample results are preserved for provenance but are not dependencies of Sift, require old tooling and proprietary data, and are not part of the supported product.

<details>
<summary>Original ArchiGAN notes</summary>

## "ArchiGAN - Artificial Architectures": PyTorch Implementation.
<!-- [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)
 -->

## Introduction

* This PyTorch implementation of ArchiGAN, a research project conducted for generative form-finding based on 3D data. Code is based off of part of the [paper](https://arxiv.org/abs/1610.07584) "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling". I provide the complete pipeline of loading dataset, training, evaluation and visualization here and also I would share some results based on different parameter settings.

### Prerequisites

* Python 3.6.5 | Anaconda
* Pytorch 0.4.1
* tensorboardX
* visdom (optional)

### Pipeline

* 3D models were converted into obj files, that were then processed using `binvox-rw.py`.

#### Data
* Data was proprietary, and needs to be placed in a folder called `volumetric_data`.

#### Training
* Then `cd src`, simply run `python main.py` on GPU or CPU. Of course, you need a GPU for training until getting good results. I used one GeForce GTX 1070 in my experiments on 3D models with resolution of 32x32x32. The maximum number of channels of feature map is 256. Because of these, the results may be inconsistent with the paper. You may need a stronger one for higher resolution one 64x64x64 and 512 feature maps. 

* Other arguments could be used, for example, `python main.py --logs=<SOMETHING_YOU_WANT_TO_LOG>` would start the tensorboardX for logging to `outputs` folder. For local debugging, you can run `python main.py --local_test=True`.

* During training, model weights and some 3D reconstruction images would be also logged to the `outputs`, `images` folders, respectively, for every `model_save_step` number of step in `params.py`. You can play with all parameters in `params.py`.

#### Evaluation
* For evaluation for trained model, you can run `python main.py --test=True` to call `tester.py`.
* If you want to visualize using visdom, first run `python -m visdom.server`, then `python main.py --test=True --use_visdom=True`.
* For more results, see the following or the `results` folder.

<!-- 
### GAN Trick
I use some more trick for better result
* the loss function to optimize G is `min (log 1-D)`, but in practice folks practically use `max log D`
* Z is Sampled from a gaussian distribution [0, 0.33]
* Use Soft Labels - It make loss function smoothing (When I don't use soft labels , I observe divergence after 500 epochs)
* learning rate scheduler - after 500 epoch, descriminator's learning rate is decayed

If you want to know more trick , 
go to  [Soumith’s ganhacks repo.](https://github.com/soumith/ganhacks)
 -->

### Basic Parameter Settings
* Here I list some basic parameter settings and in the results section I would change some specific parameters and see what happens.
* Batch size is 32, which depends on the memory and I do not see much difference by changing it.
* Learning rate, beta values for Adam and LeakyReLU parameters are the same with the original paper, as well as discriminator update trick based on accuracy.
* Latent z vector is sampled from normal(0, 0.33) following [ganhacks](https://github.com/soumith/ganhacks), but I do not use soft labels in the basic setting.
* Sigmoid function is used at both generator and discriminator for final outputs.


### Acknowledgements

* This code is a heavily modified version based on both [3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch) and [tf-3dgan](https://github.com/meetshah1995/tf-3dgan) and thanks for them.

</details>



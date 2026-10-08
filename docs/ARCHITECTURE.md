# Architecture and data contracts

This document distinguishes the **current v1 implementation** from the **target MVP contracts**. Target shapes are not implemented merely because they are documented here; `STATUS.md` and `TASKS.md` determine implementation state.

## System shape

```text
Browser
  Studio state ── React Flow canvas
       │          R3F preview/export
       ├── storage adapter ── IndexedDB
       └── /api/* ── provider adapter ── Meshy (optional)

Legacy PyTorch prototype (preserved, disconnected)
```

The browser owns local project state and procedural generation. Next.js route handlers are the only code allowed to read provider credentials or call paid hosted APIs.

## Source boundaries

- `src/app/`: routes, global styles, and server API surface.
- `src/components/studio/`: interactive shell, graph nodes, and 3D preview.
- `src/lib/contracts.ts`: versioned project and API schemas.
- `src/lib/massing.ts`: deterministic prompt-to-geometry domain logic.
- `src/lib/storage.ts`: persistence adapter; the only IndexedDB dependency boundary.
- `src/lib/providers/`: hosted-provider server code.
- `docs/`: durable coordination state.
- Root Python files and `pytorch_gan.ipynb`: preserved legacy research prototype, not a runtime dependency.

## Project contract

`SiftProject` is versioned with `schemaVersion: 1` and contains:

- stable `id`, user-facing `name`, and ISO timestamps;
- source `prompt` plus additive `refinement`;
- `provider` (`procedural` or `meshy`);
- deterministic `massing` parameters;
- graph node positions and edges;
- optional provider task metadata with status and output URL, but never credentials.

Persisted objects are parsed at the storage boundary. A future schema change must add a migration before incrementing the version.

### Target v2 aggregate

P0.09 introduces a migrated `SiftProject` aggregate with references rather than duplicated heavyweight payloads:

```ts
type SiftProjectV2 = {
  schemaVersion: 2;
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  viewport: { x: number; y: number; zoom: number };
  graph: { nodes: DesignNode[]; edges: DesignEdge[] };
  artifacts: Record<string, Artifact>;
  jobs: Record<string, GenerationJob>;
  revisions: Record<string, DesignRevision>;
  settings: ProjectSettings;
};
```

The v1-to-v2 migration must retain current prompt, refinement, massing, positions, edges, and provider task metadata. It may synthesize initial artifact/revision IDs, but it must not drop valid projects.

## Canonical target contracts

`BuildingSpec` is the single procedural source of geometry truth:

```ts
type BuildingSpec = {
  schemaVersion: 1;
  name: string;
  units: "m";
  floorHeight: number;
  footprint: { type: "rectangle"; width: number; depth: number } |
             { type: "circle"; radius: number };
  volumes: Array<{
    id: string;
    role: "podium" | "tower" | "wing" | "core";
    startFloor: number;
    floorCount: number;
    footprintScale: number;
    offsetX: number;
    offsetZ: number;
    rotationDegrees: number;
    taper: number;
    setbackEvery?: number;
    setbackAmount?: number;
    materialId: string;
  }>;
  facade: { style: "solid" | "horizontal" | "vertical" | "grid"; glazingRatio: number };
  roof: { style: "flat" | "terrace" | "crown" };
  materials: Record<string, { kind: "clay" | "concrete" | "glass" | "metal"; color: string }>;
};
```

All fields receive Zod bounds and cross-field validation, including non-overlapping floor ranges where required, positive dimensions, bounded polygon/mesh complexity, and valid material references.

Artifacts are immutable:

```ts
type Artifact = {
  id: string;
  kind: "building-spec" | "model-glb" | "render-png";
  sourceNodeId: string;
  createdAt: string;
  storageKey: string;
  metadata: Record<string, unknown>;
};
```

Jobs describe execution, not outputs:

```ts
type GenerationJob = {
  id: string;
  nodeId: string;
  provider: "procedural" | "meshy";
  providerTaskId?: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "timed-out" | "rate-limited";
  progress?: number;
  error?: { code: string; message: string; retryable: boolean };
  resultArtifactId?: string;
};
```

`DesignRevision` records `parentArtifactId`, `childArtifactId`, source node IDs, change kind (`parameters`, `prompt`, or `provider-regeneration`), user-visible instruction, and timestamp. A source artifact never changes in place.

`DesignNode` types are `prompt`, `generation`, `model`, `render`, and `variation`. Ports declare accepted/produced artifact kinds. The domain connection validator rejects incompatible edges, missing artifacts, and cycles outside an explicit iteration operation.

## Generation contract

Internal `POST /api/generate` accepts a validated prompt, refinement, and provider.

- `procedural`: returns deterministic massing parameters synchronously; no external call.
- `meshy`: when enabled, creates an asynchronous preview task and returns its task ID. The server composes the architectural prompt and sends `target_formats: ["glb"]`.

Target `GET /api/generate/:taskId` proxies normalized status for authorized project jobs. The current slice deliberately does not poll Meshy because no live account has verified the integration. Paid endpoints require an authorization boundary, rate/credit guard, and explicit user confirmation before they can be enabled beyond local development.

## Security and privacy

- `MESHY_API_KEY` is server-only and must never be serialized, logged, or stored in IndexedDB.
- Hosted generation is disabled unless both a key and `MESHY_ENABLED=true` exist.
- Treat prompts and models as user content; do not add analytics capture by default.
- Signed provider asset URLs are ephemeral. A later ingestion flow must download or ask the user to export before expiration.
- Provider task IDs and normalized status may persist; authorization headers and raw secret-bearing responses may not.
- Validate prompt length and provider values at the server boundary.

## Export contracts

- PNG is a client-side capture of the active WebGL canvas.
- GLB is created client-side from the deterministic massing group and includes geometry/materials. It is a concept model, not BIM and not guaranteed watertight or fabrication-ready.
- Hosted GLB export will use the provider output URL only after a successful, verified task.

## Testing strategy

- Unit: normalization, deterministic massing, contract validation, prompt composition.
- Component: node labels, provider states, sample loading, persistence failure messaging.
- End-to-end: create project → prompt → generate → inspect → parameter edit → branch twice → render → save/reload → GLB/PNG export.
- Provider: mocked HTTP contracts by default; live smoke test is manual and opt-in because it consumes credits.


import { hostedConfig } from "./http";
import { MAX_GLB_BYTES, ProviderError, type HostedProvider } from "./types";

// STUB — replaced by the real adapter.
const stub = (id: "hunyuan3d-rapid" | "hunyuan3d-pro", label: string): HostedProvider => ({
  id, label, costLabel: "unconfirmed", supportsCancel: true, assetHosts: ["fal.media"], maxGlbBytes: MAX_GLB_BYTES,
  config: (env) => hostedConfig(env, "HUNYUAN_ENABLED", ["FAL_KEY"]),
  create: async () => { throw new ProviderError("not-implemented", "Hunyuan3D is not implemented.", 501, false); },
  status: async () => { throw new ProviderError("not-implemented", "Hunyuan3D is not implemented.", 501, false); },
  cancel: async () => { throw new ProviderError("not-implemented", "Hunyuan3D is not implemented.", 501, false); },
});
export const hunyuanRapidProvider = stub("hunyuan3d-rapid", "Hunyuan3D Rapid");
export const hunyuanProProvider = stub("hunyuan3d-pro", "Hunyuan3D Pro");

import { hostedConfig } from "./http";
import { MAX_GLB_BYTES, ProviderError, type HostedProvider } from "./types";

// STUB — replaced by the real adapter.
export const tripoProvider: HostedProvider = {
  id: "tripo", label: "Tripo", costLabel: "unconfirmed", supportsCancel: false, assetHosts: ["tripo3d.com", "tripo3d.ai"], maxGlbBytes: MAX_GLB_BYTES,
  config: (env) => hostedConfig(env, "TRIPO_ENABLED", ["TRIPO_API_KEY"]),
  create: async () => { throw new ProviderError("not-implemented", "Tripo is not implemented.", 501, false); },
  status: async () => { throw new ProviderError("not-implemented", "Tripo is not implemented.", 501, false); },
  cancel: async () => { throw new ProviderError("not-implemented", "Tripo is not implemented.", 501, false); },
};

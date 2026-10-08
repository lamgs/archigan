import { hostedConfig } from "./http";
import { MAX_GLB_BYTES, ProviderError, type HostedProvider } from "./types";

// STUB — replaced by the real adapter.
const stub = (id: "tencent-rapid" | "tencent-pro", label: string): HostedProvider => ({
  id, label, costLabel: "unconfirmed", supportsCancel: false, assetHosts: ["tencentcos.com"], maxGlbBytes: MAX_GLB_BYTES,
  config: (env) => hostedConfig(env, "TENCENT_HY3D_ENABLED", ["TENCENT_SECRET_ID", "TENCENT_SECRET_KEY"]),
  create: async () => { throw new ProviderError("not-implemented", "Tencent is not implemented.", 501, false); },
  status: async () => { throw new ProviderError("not-implemented", "Tencent is not implemented.", 501, false); },
  cancel: async () => { throw new ProviderError("not-implemented", "Tencent is not implemented.", 501, false); },
});
export const tencentRapidProvider = stub("tencent-rapid", "HY 3D Rapid (Tencent)");
export const tencentProProvider = stub("tencent-pro", "HY 3D Pro (Tencent)");

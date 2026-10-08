import { hunyuanProProvider, hunyuanRapidProvider } from "./hunyuan";
import { meshyProvider } from "./meshy";
import { tripoProvider } from "./tripo";
import { HOSTED_PROVIDER_IDS, type HostedProvider, type HostedProviderId } from "./types";

const PROVIDERS: Record<HostedProviderId, HostedProvider> = {
  meshy: meshyProvider,
  tripo: tripoProvider,
  "hunyuan3d-rapid": hunyuanRapidProvider,
  "hunyuan3d-pro": hunyuanProProvider,
};

export const isHostedProviderId = (value: unknown): value is HostedProviderId => typeof value === "string" && (HOSTED_PROVIDER_IDS as readonly string[]).includes(value);
export const getProvider = (id: HostedProviderId): HostedProvider => PROVIDERS[id];
export const listProviders = (): HostedProvider[] => HOSTED_PROVIDER_IDS.map((id) => PROVIDERS[id]);

/** Public (secret-free) status for every provider, as served by GET /api/providers. */
export function providerCatalog(env: Record<string, string | undefined>) {
  return Object.fromEntries(listProviders().map((p) => [p.id, { label: p.label, costLabel: p.costLabel, supportsCancel: p.supportsCancel, ...p.config(env) }]));
}

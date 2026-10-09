import { tripoProvider } from "./tripo";
import { HOSTED_PROVIDER_IDS, type HostedProvider, type HostedProviderId } from "./types";

// The registry stays provider-neutral (ADR-017) even though Tripo is the only hosted provider (ADR-018).
const PROVIDERS: Record<HostedProviderId, HostedProvider> = { tripo: tripoProvider };

export const isHostedProviderId = (value: unknown): value is HostedProviderId => typeof value === "string" && (HOSTED_PROVIDER_IDS as readonly string[]).includes(value);
export const getProvider = (id: HostedProviderId): HostedProvider => PROVIDERS[id];
export const listProviders = (): HostedProvider[] => HOSTED_PROVIDER_IDS.map((id) => PROVIDERS[id]);

/** Public (secret-free) status for every provider, as served by GET /api/providers. */
export function providerCatalog(env: Record<string, string | undefined>) {
  return Object.fromEntries(listProviders().map((p) => [p.id, { label: p.label, costLabel: p.costLabel, supportsCancel: p.supportsCancel, ...p.config(env) }]));
}

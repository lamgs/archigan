import { isSupportedProvider, SUPPORTED_PROVIDERS, type Provider, type SupportedProvider } from "./contracts";

/**
 * Client-safe provider metadata (labels, rough cost, setup env var names). No secrets, no URLs. It mirrors the server
 * catalog in src/lib/providers and is only a fallback: /api/providers is preferred for label, cost and cancel support.
 *
 * Only supported providers have metadata (ADR-018). Legacy ids from old saved projects (meshy, hunyuan3d-*, tencent-*)
 * are not in PROVIDER_META; the helpers below fall back to the raw id and never throw.
 */
export { SUPPORTED_PROVIDERS, isSupportedProvider };
export type HostedProviderId = Exclude<SupportedProvider, "procedural">;
export type ProviderMeta = { id: SupportedProvider; label: string; costLabel: string; supportsCancel: boolean; enabledVar?: string; keyVar?: string };

export const PROVIDER_META: Record<SupportedProvider, ProviderMeta> = {
  procedural: { id: "procedural", label: "Local procedural", costLabel: "Free", supportsCancel: false },
  // ADR-016: about $0.28-0.35 per text-to-3D at 100 credits = $1. UNCONFIRMED estimate (keep in sync with tripo.ts).
  tripo: { id: "tripo", label: "Tripo", costLabel: "≈ $0.30 per model (estimate)", supportsCancel: false, enabledVar: "TRIPO_ENABLED", keyVar: "TRIPO_API_KEY" },
};

/** Order shown in the picker. */
export const PICKER_ORDER: SupportedProvider[] = ["procedural", "tripo"];

export type CatalogEntry = { label?: string; costLabel?: string; supportsCancel?: boolean; configured?: boolean; enabled?: boolean; hasKey?: boolean; accessCodeRequired?: boolean; verified?: boolean };
export type ProviderCatalog = Partial<Record<string, CatalogEntry>>;

const metaOf = (id: string): ProviderMeta | undefined => (isSupportedProvider(id) ? PROVIDER_META[id] : undefined);

/** Unknown or legacy ids (e.g. from an old saved job) fall back to the raw id rather than throwing. */
export const providerLabel = (id: string, catalog?: ProviderCatalog): string => catalog?.[id]?.label ?? metaOf(id)?.label ?? id;
export const providerCost = (id: string, catalog?: ProviderCatalog): string => catalog?.[id]?.costLabel ?? metaOf(id)?.costLabel ?? "unknown";
export const providerSupportsCancel = (id: string, catalog?: ProviderCatalog): boolean => catalog?.[id]?.supportsCancel ?? metaOf(id)?.supportsCancel ?? false;
export const isHostedProvider = (id: Provider | string): id is HostedProviderId => isSupportedProvider(id) && id !== "procedural";

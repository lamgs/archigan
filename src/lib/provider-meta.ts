import type { Provider } from "./contracts";

/**
 * Client-safe provider metadata (labels, rough cost, setup env var names). No secrets, no URLs. It mirrors the server
 * catalog in src/lib/providers and is only a fallback: /api/providers is preferred for label, cost and cancel support.
 */
export type HostedProviderId = Exclude<Provider, "procedural">;
export type ProviderMeta = { id: Provider; label: string; costLabel: string; supportsCancel: boolean; enabledVar?: string; keyVar?: string };

export const PROVIDER_META: Record<Provider, ProviderMeta> = {
  procedural: { id: "procedural", label: "Local procedural", costLabel: "free", supportsCancel: false },
  "hunyuan3d-rapid": { id: "hunyuan3d-rapid", label: "Hunyuan3D Rapid", costLabel: "unconfirmed", supportsCancel: true, enabledVar: "HUNYUAN_ENABLED", keyVar: "FAL_KEY" },
  "hunyuan3d-pro": { id: "hunyuan3d-pro", label: "Hunyuan3D Pro", costLabel: "unconfirmed", supportsCancel: true, enabledVar: "HUNYUAN_ENABLED", keyVar: "FAL_KEY" },
  tripo: { id: "tripo", label: "Tripo", costLabel: "unconfirmed", supportsCancel: false, enabledVar: "TRIPO_ENABLED", keyVar: "TRIPO_API_KEY" },
  meshy: { id: "meshy", label: "Meshy", costLabel: "≈ 20 credits (≈ $0.40+) — estimate, plan-dependent", supportsCancel: true, enabledVar: "MESHY_ENABLED", keyVar: "MESHY_API_KEY" },
};

/** Order shown in the picker. */
export const PICKER_ORDER: Provider[] = ["procedural", "hunyuan3d-rapid", "hunyuan3d-pro", "tripo", "meshy"];

export type CatalogEntry = { label?: string; costLabel?: string; supportsCancel?: boolean; configured?: boolean; enabled?: boolean; hasKey?: boolean; accessCodeRequired?: boolean; verified?: boolean };
export type ProviderCatalog = Partial<Record<Provider, CatalogEntry>>;

/** Unknown ids (e.g. from a newer build's saved project) fall back to the raw id rather than throwing. */
export const providerLabel = (id: string, catalog?: ProviderCatalog): string => catalog?.[id as Provider]?.label ?? PROVIDER_META[id as Provider]?.label ?? id;
export const providerCost = (id: Provider, catalog?: ProviderCatalog): string => catalog?.[id]?.costLabel ?? PROVIDER_META[id].costLabel;
export const providerSupportsCancel = (id: string, catalog?: ProviderCatalog): boolean => catalog?.[id as Provider]?.supportsCancel ?? PROVIDER_META[id as Provider]?.supportsCancel ?? false;
export const isHostedProvider = (id: Provider): id is HostedProviderId => id !== "procedural";

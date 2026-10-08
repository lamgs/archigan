"use client";

import type { Provider } from "@/lib/contracts";
import { isHostedProvider, PICKER_ORDER, PROVIDER_META, providerCost, providerLabel, type ProviderCatalog } from "@/lib/provider-meta";

/** Server env vars an operator must set to enable a hosted provider (names only; never values). */
export const setupGuidance = (id: Provider) => {
  const meta = PROVIDER_META[id];
  return meta.enabledVar ? `Set ${meta.enabledVar}=true, ${meta.keyVar} and SIFT_ACCESS_CODE on the server to enable it.` : "";
};

type Props = { provider: Provider; catalog: ProviderCatalog | null; onProvider: (provider: Provider) => void };

/** Provider choice: Local plus every hosted provider, with configured state, an "unverified" label, and an estimated cost. */
export function ProviderPicker({ provider, catalog, onProvider }: Props) {
  return (
    <fieldset className="provider-picker">
      <legend>Provider</legend>
      {PICKER_ORDER.map((id) => {
        const hosted = isHostedProvider(id);
        const configured = hosted ? Boolean(catalog?.[id]?.configured) : true;
        const disabled = !configured && provider !== id;
        const label = providerLabel(id, catalog ?? undefined);
        return (
          <label className="radio provider-option" key={id} data-provider={id}>
            <input type="radio" name="provider" checked={provider === id} disabled={disabled} onChange={() => onProvider(id)} />
            {" "}{hosted ? `${label} (paid)` : label}{" "}
            {hosted ? (
              <small>
                {configured ? "configured · unverified" : "unavailable — not configured on this server · unverified"}
                <span className="provider-option__cost"> · Estimated cost per generation: {providerCost(id, catalog ?? undefined)} (estimate, not a quote)</span>
                {!configured && <span className="provider-option__setup"> {setupGuidance(id)}</span>}
              </small>
            ) : <small>no account needed · free</small>}
          </label>
        );
      })}
    </fieldset>
  );
}

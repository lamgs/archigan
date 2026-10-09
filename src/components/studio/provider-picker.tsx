"use client";

import type { Provider } from "@/lib/contracts";
import { isHostedProvider, PICKER_ORDER, providerCost, providerLabel, type ProviderCatalog } from "@/lib/provider-meta";

type Props = { provider: Provider; catalog: ProviderCatalog | null; onProvider: (provider: Provider) => void };

/**
 * Provider choice: one identical card per provider (name, then a single muted line with the estimated cost).
 * Hosted providers carry a compact "Unverified" chip because no hosted behaviour has been tested with a real account.
 */
export function ProviderPicker({ provider, catalog, onProvider }: Props) {
  return (
    <fieldset className="provider-picker">
      <legend>Provider</legend>
      {PICKER_ORDER.map((id) => {
        const hosted = isHostedProvider(id);
        const configured = hosted ? Boolean(catalog?.[id]?.configured) : true;
        const disabled = !configured && provider !== id;
        const selected = provider === id;
        return (
          <label className={`provider-option ${selected ? "is-selected" : ""} ${disabled ? "is-disabled" : ""}`} key={id} data-provider={id}>
            <input type="radio" name="provider" checked={selected} disabled={disabled} onChange={() => onProvider(id)} />
            <span className="provider-option__text">
              <span className="provider-option__name">
                <strong>{providerLabel(id, catalog ?? undefined)}</strong>
                {hosted && <em className="chip chip--unverified">Unverified</em>}
                {hosted && !configured && <em className="chip chip--muted">Not configured on this server</em>}
              </span>
              <small className="provider-option__desc">{providerCost(id, catalog ?? undefined)}</small>
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}

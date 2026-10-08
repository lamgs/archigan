"use client";

import { useEffect, useRef } from "react";

type Props = {
  providerLabel: string;
  costLabel: string;
  supportsCancel: boolean;
  prompt: string;
  accessCode: string;
  onAccessCode: (value: string) => void;
  busy: boolean;
  error: string | null;
  onConfirm: () => void;
  onCancel: () => void;
};

/** Explicit paid-generation confirmation that names the provider before any request is made. */
export function PaidConfirm({ providerLabel, costLabel, supportsCancel, prompt, accessCode, onAccessCode, busy, error, onConfirm, onCancel }: Props) {
  const cancel = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    cancel.current?.focus();
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && !busy && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onCancel]);
  return (
    <div className="modal-backdrop" role="presentation">
      <div className="modal" role="alertdialog" aria-modal="true" aria-labelledby="paid-title" aria-describedby="paid-body">
        <h2 id="paid-title">Spend {providerLabel} credits?</h2>
        <div id="paid-body">
          <p>This sends your brief to <strong>{providerLabel}</strong>, a paid third-party service, and uses credits on the {providerLabel} account connected to this deployment. It cannot be undone, and {supportsCancel ? "a task can only be cancelled while it is still queued." : "a task cannot be cancelled once started; the app can only stop waiting."}</p>
          <p className="modal__fine">Approximate cost per generation: {costLabel} (an estimate from public pricing, not a quote).</p>
          <blockquote>{prompt}</blockquote>
          <p className="modal__fine">The result is a fixed mesh (not editable geometry). Hosted generation has not been verified against a live account yet.</p>
        </div>
        <label className="field field--stack"><span>Access code</span><input type="password" autoComplete="off" value={accessCode} onChange={(event) => onAccessCode(event.target.value)} aria-describedby="code-hint" /></label>
        <p id="code-hint" className="modal__fine">The code is held in memory only and is not saved with the project.</p>
        {error && <p role="alert" className="inspector__error">{error}</p>}
        <div className="modal__actions">
          <button ref={cancel} type="button" className="ghost-button" onClick={onCancel} disabled={busy}>Cancel</button>
          <button type="button" className="generate-button modal__go" onClick={onConfirm} disabled={busy || !accessCode.trim()}>{busy ? `Contacting ${providerLabel}…` : "Spend credits & generate"}</button>
        </div>
      </div>
    </div>
  );
}

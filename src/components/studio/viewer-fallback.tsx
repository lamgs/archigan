"use client";

import { Component, type ReactNode } from "react";

type FallbackProps = { title: string; message: string; actions?: { label: string; onClick: () => void }[]; role?: "alert" | "status" };

/** Shown in place of the 3D canvas whenever WebGL cannot be used, so the panel never silently goes blank. */
export function ViewerFallback({ title, message, actions = [], role = "alert" }: FallbackProps) {
  return (
    <div className="viewer-fallback" role={role}>
      <h3>{title}</h3>
      <p>{message}</p>
      {actions.length > 0 && <div className="viewer-fallback__actions">{actions.map((action) => <button type="button" key={action.label} onClick={action.onClick}>{action.label}</button>)}</div>}
    </div>
  );
}

type BoundaryProps = { children: ReactNode; onReset: () => void; resetKey: number };
type BoundaryState = { failed: boolean; key: number };

/** Catches errors thrown while creating or rendering the WebGL scene (context creation, shader, loader failures). */
export class ViewerBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { failed: false, key: this.props.resetKey };

  static getDerivedStateFromError(): Partial<BoundaryState> {
    return { failed: true };
  }

  static getDerivedStateFromProps(props: BoundaryProps, state: BoundaryState): Partial<BoundaryState> | null {
    return props.resetKey !== state.key ? { failed: false, key: props.resetKey } : null;
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return <ViewerFallback title="The 3D view stopped working" message="Something went wrong while drawing the model. Your project is safe. Try restarting the view; if it keeps failing, download the GLB and open it elsewhere." actions={[{ label: "Restart 3D view", onClick: this.props.onReset }]} />;
  }
}

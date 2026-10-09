"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import type { DesignNodeType } from "@/lib/contracts";
import { NODE_PORTS } from "@/lib/graph";
import { NODE_LABELS } from "@/lib/workflow";

export type StudioNodeData = {
  type: DesignNodeType;
  params: Record<string, unknown>;
  artifactId?: string;
  status: "ready" | "stale" | "pending" | "blocked";
  message: string;
  summary: string;
  nextTypes: DesignNodeType[];
  onText: (id: string, value: string) => void;
  onAdd: (sourceId: string, type: DesignNodeType) => void;
  onRun: (id: string) => void;
  onBranch: (id: string) => void;
  onCommit: () => void;
} & Record<string, unknown>;

export type StudioFlowNode = Node<StudioNodeData, "studio">;

const EYEBROW: Record<DesignNodeType, string> = { prompt: "Intent", generation: "Generate", variation: "Direct", model: "Model", render: "Render" };
const STATUS_LABEL = { ready: "Ready", stale: "Out of date", pending: "Needs render", blocked: "Waiting" } as const;

const GLYPHS: Record<DesignNodeType, React.ReactNode> = {
  prompt: <path d="M3 4h10M3 8h10M3 12h6" />,
  generation: <path d="M8 2l1.6 4.4L14 8l-4.4 1.6L8 14l-1.6-4.4L2 8l4.4-1.6z" />,
  variation: <path d="M4 3v4a3 3 0 0 0 3 3h5M4 3L2 5m2-2l2 2M12 7l2 3-2 3" />,
  model: <path d="M8 2l5 2.8v6.4L8 14l-5-2.8V4.8zM8 8.4V14M3 4.8l5 3.6 5-3.6" />,
  render: <path d="M2 5h2.5l1-1.5h5L11.5 5H14v8H2zM8 11.2a2.4 2.4 0 1 0 0-4.8 2.4 2.4 0 0 0 0 4.8z" />,
};

export function StudioNode({ id, data, selected }: NodeProps<StudioFlowNode>) {
  const ports = NODE_PORTS[data.type];
  const editable = data.type === "prompt" || data.type === "variation";
  const text = typeof data.params.text === "string" ? data.params.text : "";
  return (
    <article className={`studio-node studio-node--${data.type} ${selected ? "is-selected" : ""}`} data-status={data.status} aria-label={`${NODE_LABELS[data.type]} node`}>
      {ports.inputs.map((port, index) => (
        <Handle key={port.id} id={port.id} type="target" position={Position.Left} className={`port port--${port.kind}`} title={`Input: ${port.kind}`} style={{ top: `${50 + index * 18}%` }} />
      ))}
      <div className="studio-node__topline">
        <i className="studio-node__glyph" aria-hidden="true"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">{GLYPHS[data.type]}</svg></i>
        <span>{EYEBROW[data.type]}</span>
        <em className={`status status--${data.status}`} title={data.message}>{STATUS_LABEL[data.status]}</em>
      </div>
      <h2>{NODE_LABELS[data.type]}{typeof data.params.label === "string" && data.params.label ? <small className="node-label"> · {data.params.label}</small> : null}</h2>
      {editable ? (
        <textarea
          className="nodrag nowheel"
          aria-label={data.type === "prompt" ? "Design brief" : "Refinement"}
          value={text}
          rows={3}
          maxLength={data.type === "prompt" ? 800 : 400}
          placeholder={data.type === "prompt" ? "Describe the building, material, and organization…" : "Optional: terraces, twist, glass…"}
          onChange={(event) => data.onText(id, event.target.value)}
          onBlur={data.onCommit}
        />
      ) : (
        <p>{data.status === "blocked" || data.status === "pending" ? data.message : data.summary}</p>
      )}
      {data.type === "generation" && <button type="button" className="nodrag node-run" onClick={() => data.onRun(id)}>{data.artifactId ? "Run again" : "Run"}</button>}
      {data.type === "render" && data.status !== "blocked" && <button type="button" className="nodrag node-run" onClick={() => data.onRun(id)}>{data.artifactId ? "Render again" : "Render"}</button>}
      {selected && (data.type === "generation" || data.type === "variation") && data.status !== "blocked" && <button type="button" className="nodrag node-branch" onClick={() => data.onBranch(id)}>Branch ⑂</button>}
      {data.status === "stale" && <p className="node-note">The upstream prompt changed — run again to update.</p>}
      {selected && data.nextTypes.length > 0 && (
        <div className="node-next nodrag" role="group" aria-label="Add next node">
          <span>Add next</span>
          {data.nextTypes.map((type) => <button type="button" key={type} onClick={() => data.onAdd(id, type)}>{NODE_LABELS[type]}</button>)}
        </div>
      )}
      {ports.outputs.map((port, index) => (
        <Handle key={port.id} id={port.id} type="source" position={Position.Right} className={`port port--${port.kind}`} title={`Output: ${port.kind}`} style={{ top: `${50 + index * 18}%` }} />
      ))}
    </article>
  );
}

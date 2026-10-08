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

export function StudioNode({ id, data, selected }: NodeProps<StudioFlowNode>) {
  const ports = NODE_PORTS[data.type];
  const editable = data.type === "prompt" || data.type === "variation";
  const text = typeof data.params.text === "string" ? data.params.text : "";
  return (
    <article className={`studio-node studio-node--${data.type} ${selected ? "is-selected" : ""}`} aria-label={`${NODE_LABELS[data.type]} node`}>
      {ports.inputs.map((port, index) => (
        <Handle key={port.id} id={port.id} type="target" position={Position.Left} className={`port port--${port.kind}`} title={`Input: ${port.kind}`} style={{ top: `${50 + index * 18}%` }} />
      ))}
      <div className="studio-node__topline">
        <span>{EYEBROW[data.type]}</span>
        <em className={`status status--${data.status}`} title={data.message}>{STATUS_LABEL[data.status]}</em>
      </div>
      <h3>{NODE_LABELS[data.type]}{typeof data.params.label === "string" && data.params.label ? <small className="node-label"> · {data.params.label}</small> : null}</h3>
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

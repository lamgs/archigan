"use client";

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export type StudioNodeData = {
  kind: "brief" | "massing" | "refine" | "export";
  eyebrow: string;
  title: string;
  body: string;
  meta: string;
};

export type StudioFlowNode = Node<StudioNodeData, "studio">;

export function StudioNode({ data, selected }: NodeProps<StudioFlowNode>) {
  return (
    <article className={`studio-node studio-node--${data.kind} ${selected ? "is-selected" : ""}`}>
      {data.kind !== "brief" && <Handle type="target" position={Position.Left} />}
      <div className="studio-node__topline">
        <span>{data.eyebrow}</span>
        <i aria-hidden="true" />
      </div>
      <h3>{data.title}</h3>
      <p>{data.body}</p>
      <footer>{data.meta}</footer>
      {data.kind !== "export" && <Handle type="source" position={Position.Right} />}
    </article>
  );
}


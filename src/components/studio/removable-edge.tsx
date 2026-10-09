"use client";

import { BaseEdge, EdgeLabelRenderer, getBezierPath, type Edge, type EdgeProps } from "@xyflow/react";

export type RemovableEdgeData = { onRemove?: (edgeId: string) => void } & Record<string, unknown>;
export type RemovableFlowEdge = Edge<RemovableEdgeData, "removable">;

/** A bezier edge that shows a labelled × button at its midpoint while it is selected. */
export function RemovableEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd, selected, data }: EdgeProps<RemovableFlowEdge>) {
  const [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  return (
    <>
      <BaseEdge id={id} path={path} style={style} markerEnd={markerEnd} />
      {selected && (
        <EdgeLabelRenderer>
          <button type="button" className="edge-remove nodrag nopan" aria-label="Remove connection" title="Remove connection (Delete)" style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }} onClick={() => data?.onRemove?.(id)}>×</button>
        </EdgeLabelRenderer>
      )}
    </>
  );
}
